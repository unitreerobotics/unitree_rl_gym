#ifndef DATA_BUFFER_H
#define DATA_BUFFER_H

#include <memory>
#include "AtomicLock.h"

template<typename T>
class DataBuffer
{
	public:
		explicit DataBuffer() = default;
		~DataBuffer() = default;

		void SetDataPtr(const std::shared_ptr<T>& dataPtr)
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr = dataPtr;
		}

		std::shared_ptr<T> GetDataPtr(bool clear = false)
		{
			ScopedLock<AFLock> lock(mLock);
			if (clear)
			{
				std::shared_ptr<T> dataPtr = mDataPtr;
				mDataPtr.reset();
				return dataPtr;
			}
			else
			{
				return mDataPtr;
			}
		}

		void SwapDataPtr(std::shared_ptr<T>& dataPtr)
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr.swap(dataPtr);
		}

		void SetData(const T& data)
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr = std::shared_ptr<T>(new T(data));
		}

		/*
		 * Not recommend because an additional data assignment.
		 */
		bool GetData(T& data, bool clear = false)
		{
			ScopedLock<AFLock> lock(mLock);
			if (mDataPtr == NULL)
			{
				return false;
			}

			data = *mDataPtr;
			if (clear)
			{
				mDataPtr.reset();
			}

			return true;
		}

		void Clear()
		{
			ScopedLock<AFLock> lock(mLock);
			mDataPtr.reset();
		}

	private:
		std::shared_ptr<T> mDataPtr;
		AFLock mLock;
};
#endif
