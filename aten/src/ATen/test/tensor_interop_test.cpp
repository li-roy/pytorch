#include "gtest/gtest.h"

#include "ATen/ATen.h"
#include "caffe2/core/tensor.h"

TEST(TestTensorInterop, SimpleCaffe2ToPytorch) {
  caffe2::Tensor c2_tensor(caffe2::CPU);
  c2_tensor.Resize(4, 4);
  auto data = c2_tensor.mutable_data<std::string>();
  for (int64_t i = 0; i < 16; i++) {
    data[i] = std::to_string(i);
  }
  at::Tensor at_tensor(c2_tensor.getIntrusivePtr());
  at::TensorImpl* impl = at_tensor.unsafeGetTensorImpl();

  auto it = impl->data<std::string>();
  for (int64_t i = 0; i < 16; i++) {
    ASSERT_EQ(it[i], std::to_string(i));
  }
}

TEST(TestTensorInterop, SimplePytorchToCaffe2) {
  auto at_tensor = at::ones({5, 5}, at::dtype(at::kLong));
  caffe2::Tensor c2_tensor(at_tensor.getIntrusivePtr());
  at::TensorImpl* impl = at_tensor.unsafeGetTensorImpl();

  auto it = impl->data<int64_t>();
  for (int64_t i = 0; i < 25; i++) {
    ASSERT_EQ(it[i], 1);
  }
}
