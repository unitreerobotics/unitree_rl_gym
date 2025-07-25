#ifndef ATOMIC_LOCK_H
#define ATOMIC_LOCK_H

#include <atomic>

class AFLock
{
	public:
		AFLock() = default;
		~AFLock() = default;

		void Lock()
		{
			while (mAFL.test_and_set(std::memory_order_acquire));
		}

		bool TryLock()
		{
			return mAFL.test_and_set(std::memory_order_acquire) ? false : true;
		}

		void Unlock()
		{
			mAFL.clear(std::memory_order_release);
		}

	private:
		std::atomic_flag mAFL = ATOMIC_FLAG_INIT;
};

template<typename L>
class ScopedLock
{
	public:
		explicit ScopedLock(L& lock) :
			mLock(lock)
	{
		mLock.Lock();
	}

		~ScopedLock()
		{
			mLock.Unlock();
		}

	private:
		L& mLock;
};
#endif
