/**
 * @file parallel_for.hpp
 * @brief Minimal intra-rank parallel executor for CPU backends.
 *
 * ParaFaFT parallelises each stage by splitting the *batch* of independent 1D
 * transforms across threads, rather than relying on FFTW's internal threading
 * (which parallelises *within* a transform). Splitting the batch is
 * embarrassingly parallel and generally scales better; it also lets us place
 * pages NUMA-locally via a matching first-touch pass.
 *
 * Three implementations are selected at configure time:
 * - PARAFAFT_PARALLEL_OMP     — OpenMP parallel region (preferred when found)
 * - PARAFAFT_PARALLEL_THREADS — persistent std::thread pool (POSIX threads on
 *                               Linux); used when OpenMP is unavailable
 * - neither                   — serial; run() invokes the body once with tid 0
 *
 * The callable receives (thread_index, thread_count) and must not throw: the
 * pool has no cross-thread exception propagation and a escaping exception
 * terminates the process.
 */

#ifndef PARAFAFT_PARALLEL_FOR_HPP
#define PARAFAFT_PARALLEL_FOR_HPP

#include <cstddef>

#if defined(PARAFAFT_PARALLEL_OMP)
#include <omp.h>
#elif defined(PARAFAFT_PARALLEL_THREADS)
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#endif

namespace parafaft
{
  namespace detail
  {
    /**
     * @brief Split [0, total) into `nthreads` contiguous blocks.
     *
     * Block `tid` receives `total/nthreads` items, with the first
     * `total%nthreads` blocks taking one extra, so every item is covered
     * exactly once and sizes differ by at most one.
     *
     * @param total     Number of items to distribute.
     * @param tid       Block index in [0, nthreads).
     * @param nthreads  Number of blocks.
     * @param begin     [out] First item index of this block.
     * @param end       [out] One-past-last item index of this block.
     */
    inline void chunk_range(std::size_t total, int tid, int nthreads, std::size_t &begin,
                            std::size_t &end)
    {
      const std::size_t t = static_cast<std::size_t>(tid);
      const std::size_t n = static_cast<std::size_t>(nthreads);
      const std::size_t quot = total / n;
      const std::size_t rem = total % n;
      begin = t * quot + (t < rem ? t : rem);
      end = begin + quot + (t < rem ? 1 : 0);
    }

    /**
     * @brief Runs a callable once per thread, blocking until all have finished.
     *
     * Move-only. Construct with the desired thread count; a count <= 1 makes
     * run() a direct call on the calling thread with no synchronisation.
     */
    class ParallelExecutor
    {
    public:
      explicit ParallelExecutor(int nthreads) : nthreads_(nthreads > 0 ? nthreads : 1)
      {
#if defined(PARAFAFT_PARALLEL_THREADS)
        // Worker i handles thread index i+1; the calling thread handles index 0.
        for (int i = 1; i < nthreads_; ++i) {
          workers_.emplace_back(&ParallelExecutor::worker_loop, this, i);
        }
#endif
      }

      ~ParallelExecutor()
      {
#if defined(PARAFAFT_PARALLEL_THREADS)
        {
          std::lock_guard<std::mutex> lock(mutex_);
          stop_ = true;
        }
        start_cv_.notify_all();
        for (std::thread &w : workers_) {
          if (w.joinable()) w.join();
        }
#endif
      }

      ParallelExecutor(const ParallelExecutor &) = delete;
      ParallelExecutor &operator=(const ParallelExecutor &) = delete;

      /// @brief Number of threads this executor dispatches to.
      int size() const { return nthreads_; }

      /**
       * @brief Invoke `func(thread_index, thread_count)` on every thread.
       *
       * Returns once every invocation has completed.
       */
      template <typename Func>
      void run(Func &&func)
      {
        if (nthreads_ <= 1) {
          func(0, 1);
          return;
        }
#if defined(PARAFAFT_PARALLEL_OMP)
        // `func` is shared by reference; each thread reads its own index.
#pragma omp parallel num_threads(nthreads_)
        {
          func(omp_get_thread_num(), omp_get_num_threads());
        }
#elif defined(PARAFAFT_PARALLEL_THREADS)
        // Wrap on the stack so workers can call back without type erasure that
        // allocates. `task` outlives the workers' use of it: run() does not
        // return until pending_ reaches zero.
        const TaskImpl<Func> task(func, nthreads_);
        {
          std::lock_guard<std::mutex> lock(mutex_);
          task_ = &task;
          pending_ = nthreads_ - 1;
          ++generation_;
        }
        start_cv_.notify_all();

        func(0, nthreads_);

        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return pending_ == 0; });
        task_ = nullptr;
#else
        func(0, 1);
#endif
      }

    private:
      int nthreads_;

#if defined(PARAFAFT_PARALLEL_THREADS)
      /// Type-erased view of the caller's lambda, stack-allocated by run().
      struct Task {
        virtual void invoke(int tid) const = 0;

      protected:
        ~Task() = default;
      };

      template <typename Func>
      struct TaskImpl final : Task {
        TaskImpl(Func &func, int nthreads) : func_(func), nthreads_(nthreads) {}
        void invoke(int tid) const override { func_(tid, nthreads_); }

      private:
        Func &func_;
        int nthreads_;
      };

      void worker_loop(int tid)
      {
        unsigned long long seen = 0;
        for (;;) {
          const Task *task = nullptr;
          {
            std::unique_lock<std::mutex> lock(mutex_);
            start_cv_.wait(lock, [this, seen] { return stop_ || generation_ != seen; });
            if (stop_) return;
            seen = generation_;
            task = task_;
          }

          task->invoke(tid);

          {
            std::lock_guard<std::mutex> lock(mutex_);
            if (--pending_ == 0) done_cv_.notify_one();
          }
        }
      }

      std::vector<std::thread> workers_;
      std::mutex mutex_;
      std::condition_variable start_cv_;
      std::condition_variable done_cv_;
      const Task *task_ = nullptr;
      unsigned long long generation_ = 0;
      int pending_ = 0;
      bool stop_ = false;
#endif
    };

  } // namespace detail
} // namespace parafaft

#endif // PARAFAFT_PARALLEL_FOR_HPP
