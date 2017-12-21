#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;   //blob最大维数

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>

//默认构造函数
class Blob {
 public:
  Blob()
       : data_(), diff_(), count_(0), capacity_(0) {}

  /// @brief Deprecated; use <code>Blob(const vector<int>& shape)</code>.
  explicit Blob(const int num, const int channels, const int height,
      const int width);
  explicit Blob(const vector<int>& shape);

  /// @brief Deprecated; use <code>Reshape(const vector<int>& shape)</code>.
  bool Reshape(const int num, const int channels, const int height,
      const int width);
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  bool Reshape(const vector<int>& shape);
  bool Reshape(const BlobShape& shape);
  bool ReshapeLike(const Blob& other);

  //将shape转化为string
  inline string shape_string() const {
    ostringstream stream;
    for (int i = 0; i < shape_.size(); ++i) {
      stream << shape_[i] << " ";
    }
    stream << "(" << count_ << ")";
    return stream.str();
  }
  inline const vector<int>& shape() const { return shape_; }
  /**
   * @brief Returns the dimension of the index-th axis (or the negative index-th
   *        axis from the end, if index is negative).
   *
   * @param index the axis index, which may be negative as it will be
   *        "canonicalized" using CanonicalAxisIndex.
   *        Dies on out of range index.
   */

  //获取某一维的尺寸
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  //获取维数
  inline int num_axes() const { return shape_.size(); }

  //获取数据大小
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */

  //获取start_axis到end_axis维数据的大小
  inline int count(int start_axis, int end_axis) const {
    CHECK_LE(start_axis, end_axis);
    CHECK_GE(start_axis, 0);
    CHECK_GE(end_axis, 0);
    CHECK_LE(start_axis, num_axes());
    CHECK_LE(end_axis, num_axes());
    int count = 1;
    for (int i = start_axis; i < end_axis; ++i) {
      count *= shape(i);
    }
    return count;
  }
  /**
   * @brief Compute the volume of a slice spanning from a particular first
   *        axis to the final axis.
   *
   * @param start_axis The first axis to include in the slice.
   */

  //获取start_axis到结束时的数据的大小
  inline int count(int start_axis) const {
    return count(start_axis, num_axes());
  }

  /**
   * @brief Returns the 'canonical' version of a (usually) user-specified axis,
   *        allowing for negative indexing (e.g., -1 for the last axis).
   *
   * @param axis_index the axis index.
   *        If 0 <= index < num_axes(), return index.
   *        If -num_axes <= index <= -1, return (num_axes() - (-index)),
   *        e.g., the last axis index (num_axes() - 1) if index == -1,
   *        the second to last if index == -2, etc.
   *        Dies on out of range index.
   */

  //处理下标正负,Blob的Index是可以从负坐标开始读的,标准化索引，主要是对参数索引进行标准化，以满足要求，转换坐标轴索引[-N，N]为[0，N] 
  inline int CanonicalAxisIndex(int axis_index) const {
    CHECK_GE(axis_index, -num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    CHECK_LT(axis_index, num_axes())
        << "axis " << axis_index << " out of range for " << num_axes()
        << "-D Blob with shape " << shape_string();
    if (axis_index < 0) {
      return axis_index + num_axes();
    }
    return axis_index;
  }


  ///Blob中的4个基本变量num,channel,height,width可以直接通过shape(0),shape(1),shape(2),shape(3)来访问 
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }

  //data_维数不大于4时才能使用，功能同shape()类似。 
  inline int LegacyShape(int index) const {
    CHECK_LE(num_axes(), 4)
        << "Cannot use legacy accessors on Blobs with > 4 axes.";
    CHECK_LT(index, 4);
    CHECK_GE(index, -4);
    if (index >= num_axes() || index < -num_axes()) {
      // Axis is out of range, but still in [0, 3] (or [-4, -1] for reverse
      // indexing) -- this special case simulates the one-padding used to fill
      // extraneous axes of legacy blobs.
      return 1;
    }
    return shape(index);
  }

  //计算offset,offset计算的方式也支持两种方式，一种直接指定n,c,h,w或者放到一个vector中进行计算，  
  //偏移量是根据对应的n,c,h,w，返回的offset是((n*channels()+c)*height()+h)*width()+w  
  inline int offset(const int n, const int c = 0, const int h = 0,
      const int w = 0) const {
    CHECK_GE(n, 0);
    CHECK_LE(n, num());
    CHECK_GE(channels(), 0);
    CHECK_LE(c, channels());
    CHECK_GE(height(), 0);
    CHECK_LE(h, height());
    CHECK_GE(width(), 0);
    CHECK_LE(w, width());
    return ((n * channels() + c) * height() + h) * width() + w;
  }

  //indices中存储的就是n,c,h,w
  inline int offset(const vector<int>& indices) const {
    CHECK_LE(indices.size(), num_axes());
    int offset = 0;
    for (int i = 0; i < num_axes(); ++i) {
      offset *= shape(i);
      if (indices.size() > i) {
        CHECK_GE(indices[i], 0);
        CHECK_LT(indices[i], shape(i));
        offset += indices[i];
      }
    }
    return offset;
  }
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param copy_diff if false, copy the data; if true, copy the diff
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */

   //按值拷贝blob到当前blob。一个blob中copy数据 ，通过开关控制是否copy_diff,如果是False则copy data。reshape控制是否需要reshape  
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  /*这一部分函数主要通过给定的位置访问数据，根据位置计算与数据起始
  的偏差offset，在通过cpu_data*指针获得地址
  */
  //获取某位置的data_数据  
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  //获取某位置的diff_数据  
  inline Dtype diff_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_diff()[offset(n, c, h, w)];
  }

  inline Dtype data_at(const vector<int>& index) const {
    return cpu_data()[offset(index)];
  }

  inline Dtype diff_at(const vector<int>& index) const {
    return cpu_diff()[offset(index)];
  }

  ////获取data_  
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  //获取diff
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  //这里有data和diff两类数据，而这个diff就是我们所熟知的偏差，前者主要存储  
  //前向传递的数据，而后者存储的是反向传播中的梯度  
  const Dtype* cpu_data() const;//只读获取data_ cpu指针  
  void set_cpu_data(Dtype* data);//设置data_的cpu指针，只是修改了指针  
  void set_cpu_diff(Dtype* data);
  void set_gpu_data(Dtype* data);  
  void set_gpu_diff(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;//获取data_的gpu指针
  const Dtype* cpu_diff() const; //获取diff_的cpu指针
  const Dtype* gpu_diff() const;//获取diff_的gpu指针  
  Dtype* mutable_cpu_data();//见SyncedMemory的mutable_cpu_data()，mutable是可读写访问  
  Dtype* mutable_gpu_data();//见SyncedMemory的mutable_gpu_data();  
  Dtype* mutable_cpu_diff();//见SyncedMemory的mutable_cpu_data();  
  Dtype* mutable_gpu_diff();//见SyncedMemory的mutable_gpu_data();  

  //更新data_的数据,减去diff_的数据，就是合并data和diff  
  void Update();

  /*
  其中用到math_functions.hpp中的函数caffe_axpy(),该函数封装了cblas_saxpy，实现的是Y=alpha*X+Y。
  由此，知该函数的功能是data_=(data_-diff_)。另外，该函数只实现了对double和float型数据，
  对于unsigned int和int由于该函数主要是在Net中被调用，只有Blob<float>和Blob<double>型式，
  因此没有定义unsigned int和int。从proto中恢复一个blob对象
  */
  void FromProto(const BlobProto& proto, bool reshape = true);

  /*
  由BlobProto对Blob进行赋值操作。reshape代表是否允许修改shape_的大小。
  需要注意的是再这里有double和float两种类型的数据 ，将blob序列化为proto，在代码中可以看到具体的体现
  */
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /*
  功能：计算L1范数,向量各个元素绝对值之和
  说明：其中用到了math_function.hpp中的函数caffe_cpu_asum()和caffe_gpu_asum，实现的功能是对向量X求其每个元素绝对值的和，不同的是X分别在cpu和gpu中。
  */
  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;

  /*
  功能：计算L2范数,向量各个元素的平方求和然后求平方根
  说明：用到了math_function.hpp中的caffe_cpu_dot(),caffe_cpu_strided_dot(),caffe_gpu_dot(), caffe_gpu_strided_dot()。具体就是就向量X的平方和。
  */
  /// @brief Compute the sum of squares (L2 norm squared) of the data.
  Dtype sumsq_data() const;
  /// @brief Compute the sum of squares (L2 norm squared) of the diff.
  Dtype sumsq_diff() const;

  void Clamp(Dtype lower_bound, Dtype upper_bound);

  /// @brief Scale the blob data by a constant factor.
  void scale_data(Dtype scale_factor);
  /// @brief Scale the blob diff by a constant factor.
  void scale_diff(Dtype scale_factor);

  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);//本Blob共享other的data_  
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);//本Blob共享other的diff_  

  bool ShapeEquals(const BlobProto& other);//判断other与本Blob形状是否相同。  

 protected:

  //data_指针，指针类型是shared_ptr，属于boost库的一个智能指针，这一部分主要用来申请内存存储data，data主要是正向传播的时候用的  
  shared_ptr<SyncedMemory> data_;
  
  //diff_主要用来存储偏差，update data,偏置
  shared_ptr<SyncedMemory> diff_;
  
  //智能指针存储shape_,即Blob形状,为啥要重新顶一个?优化?
  shared_ptr<SyncedMemory> shape_data_;

  //shape_存储Blob的形状  
  vector<int> shape_;

  //count_表示Blob中的元素个数，也就是个数*通道数*高度*宽度 
  int count_;

  //capacity表示当前的元素个数，因为Blob可能会reshape  
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
