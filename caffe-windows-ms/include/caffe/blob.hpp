#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;   //blob���ά��

namespace caffe {

/**
 * @brief A wrapper around SyncedMemory holders serving as the basic
 *        computational unit through which Layer%s, Net%s, and Solver%s
 *        interact.
 *
 * TODO(dox): more thorough description.
 */
template <typename Dtype>

//Ĭ�Ϲ��캯��
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

  //��shapeת��Ϊstring
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

  //��ȡĳһά�ĳߴ�
  inline int shape(int index) const {
    return shape_[CanonicalAxisIndex(index)];
  }

  //��ȡά��
  inline int num_axes() const { return shape_.size(); }

  //��ȡ���ݴ�С
  inline int count() const { return count_; }

  /**
   * @brief Compute the volume of a slice; i.e., the product of dimensions
   *        among a range of axes.
   *
   * @param start_axis The first axis to include in the slice.
   *
   * @param end_axis The first axis to exclude from the slice.
   */

  //��ȡstart_axis��end_axisά���ݵĴ�С
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

  //��ȡstart_axis������ʱ�����ݵĴ�С
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

  //�����±�����,Blob��Index�ǿ��ԴӸ����꿪ʼ����,��׼����������Ҫ�ǶԲ����������б�׼����������Ҫ��ת������������[-N��N]Ϊ[0��N] 
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


  ///Blob�е�4����������num,channel,height,width����ֱ��ͨ��shape(0),shape(1),shape(2),shape(3)������ 
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  inline int num() const { return LegacyShape(0); }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return LegacyShape(1); }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return LegacyShape(2); }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return LegacyShape(3); }

  //data_ά��������4ʱ����ʹ�ã�����ͬshape()���ơ� 
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

  //����offset,offset����ķ�ʽҲ֧�����ַ�ʽ��һ��ֱ��ָ��n,c,h,w���߷ŵ�һ��vector�н��м��㣬  
  //ƫ�����Ǹ��ݶ�Ӧ��n,c,h,w�����ص�offset��((n*channels()+c)*height()+h)*width()+w  
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

  //indices�д洢�ľ���n,c,h,w
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

   //��ֵ����blob����ǰblob��һ��blob��copy���� ��ͨ�����ؿ����Ƿ�copy_diff,�����False��copy data��reshape�����Ƿ���Ҫreshape  
  void CopyFrom(const Blob<Dtype>& source, bool copy_diff = false,
      bool reshape = false);

  /*��һ���ֺ�����Ҫͨ��������λ�÷������ݣ�����λ�ü�����������ʼ
  ��ƫ��offset����ͨ��cpu_data*ָ���õ�ַ
  */
  //��ȡĳλ�õ�data_����  
  inline Dtype data_at(const int n, const int c, const int h,
      const int w) const {
    return cpu_data()[offset(n, c, h, w)];
  }

  //��ȡĳλ�õ�diff_����  
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

  ////��ȡdata_  
  inline const shared_ptr<SyncedMemory>& data() const {
    CHECK(data_);
    return data_;
  }

  //��ȡdiff
  inline const shared_ptr<SyncedMemory>& diff() const {
    CHECK(diff_);
    return diff_;
  }

  //������data��diff�������ݣ������diff������������֪��ƫ�ǰ����Ҫ�洢  
  //ǰ�򴫵ݵ����ݣ������ߴ洢���Ƿ��򴫲��е��ݶ�  
  const Dtype* cpu_data() const;//ֻ����ȡdata_ cpuָ��  
  void set_cpu_data(Dtype* data);//����data_��cpuָ�룬ֻ���޸���ָ��  
  void set_cpu_diff(Dtype* data);
  void set_gpu_data(Dtype* data);  
  void set_gpu_diff(Dtype* data);
  const int* gpu_shape() const;
  const Dtype* gpu_data() const;//��ȡdata_��gpuָ��
  const Dtype* cpu_diff() const; //��ȡdiff_��cpuָ��
  const Dtype* gpu_diff() const;//��ȡdiff_��gpuָ��  
  Dtype* mutable_cpu_data();//��SyncedMemory��mutable_cpu_data()��mutable�ǿɶ�д����  
  Dtype* mutable_gpu_data();//��SyncedMemory��mutable_gpu_data();  
  Dtype* mutable_cpu_diff();//��SyncedMemory��mutable_cpu_data();  
  Dtype* mutable_gpu_diff();//��SyncedMemory��mutable_gpu_data();  

  //����data_������,��ȥdiff_�����ݣ����Ǻϲ�data��diff  
  void Update();

  /*
  �����õ�math_functions.hpp�еĺ���caffe_axpy(),�ú�����װ��cblas_saxpy��ʵ�ֵ���Y=alpha*X+Y��
  �ɴˣ�֪�ú����Ĺ�����data_=(data_-diff_)�����⣬�ú���ֻʵ���˶�double��float�����ݣ�
  ����unsigned int��int���ڸú�����Ҫ����Net�б����ã�ֻ��Blob<float>��Blob<double>��ʽ��
  ���û�ж���unsigned int��int����proto�лָ�һ��blob����
  */
  void FromProto(const BlobProto& proto, bool reshape = true);

  /*
  ��BlobProto��Blob���и�ֵ������reshape�����Ƿ������޸�shape_�Ĵ�С��
  ��Ҫע�������������double��float�������͵����� ����blob���л�Ϊproto���ڴ����п��Կ������������
  */
  void ToProto(BlobProto* proto, bool write_diff = false) const;

  /*
  ���ܣ�����L1����,��������Ԫ�ؾ���ֵ֮��
  ˵���������õ���math_function.hpp�еĺ���caffe_cpu_asum()��caffe_gpu_asum��ʵ�ֵĹ����Ƕ�����X����ÿ��Ԫ�ؾ���ֵ�ĺͣ���ͬ����X�ֱ���cpu��gpu�С�
  */
  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  /// @brief Compute the sum of absolute values (L1 norm) of the diff.
  Dtype asum_diff() const;

  /*
  ���ܣ�����L2����,��������Ԫ�ص�ƽ�����Ȼ����ƽ����
  ˵�����õ���math_function.hpp�е�caffe_cpu_dot(),caffe_cpu_strided_dot(),caffe_gpu_dot(), caffe_gpu_strided_dot()��������Ǿ�����X��ƽ���͡�
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
  void ShareData(const Blob& other);//��Blob����other��data_  
  /**
   * @brief Set the diff_ shared_ptr to point to the SyncedMemory holding the
   *        diff_ of Blob other -- useful in Layer%s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's diff_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareDiff(const Blob& other);//��Blob����other��diff_  

  bool ShapeEquals(const BlobProto& other);//�ж�other�뱾Blob��״�Ƿ���ͬ��  

 protected:

  //data_ָ�룬ָ��������shared_ptr������boost���һ������ָ�룬��һ������Ҫ���������ڴ�洢data��data��Ҫ�����򴫲���ʱ���õ�  
  shared_ptr<SyncedMemory> data_;
  
  //diff_��Ҫ�����洢ƫ�update data,ƫ��
  shared_ptr<SyncedMemory> diff_;
  
  //����ָ��洢shape_,��Blob��״,ΪɶҪ���¶�һ��?�Ż�?
  shared_ptr<SyncedMemory> shape_data_;

  //shape_�洢Blob����״  
  vector<int> shape_;

  //count_��ʾBlob�е�Ԫ�ظ�����Ҳ���Ǹ���*ͨ����*�߶�*���� 
  int count_;

  //capacity��ʾ��ǰ��Ԫ�ظ�������ΪBlob���ܻ�reshape  
  int capacity_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_