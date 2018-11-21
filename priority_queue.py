import tensorflow as tf


def get_n_most_salient_regions_of_size_m_by_m(image, n, m):
  """Gets the top n most salient regions of the image of size m by m

  https://stackoverflow.com/questions/52837433/extract-max-sum-submatrices

  The most salient region of the image is defined as max sum

  Returns:
    The top left index of each sub-matrix
  """
  input = tf.placeholder(tf.int32, [None, None])
  submatrix_dims = tf.placeholder(tf.int32, [2])
  numer_of_top_submatrices = tf.placeholder(tf.int32, [])

  input_shape = tf.shape(input)
  rows, cols = input_shape[0], input_shape[1]
  submatrix_rows, submatrix_cols = submatrix_dims[0], submatrix_dims[1]
  subm_rows, subm_cols = rows - submatrix_rows + 1, cols - submatrix_cols + 1

  ii, jj = tf.meshgrid(tf.range(subm_rows), tf.range(subm_cols), indexing='ij')
  d_ii, d_jj = tf.meshgrid(tf.range(submatrix_rows), tf.range(submatrix_cols),
                           indexing='ij')

  subm_ii = ii[:, :, tf.newaxis, tf.newaxis] + d_ii
  subm_jj = jj[:, :, tf.newaxis, tf.newaxis] + d_jj

  submatrices_tensor = tf.gather_nd(input, tf.stack([subm_ii, subm_jj],
                                                    axis=-1))

  submatrices_sum = tf.reduce_sum(submatrices_tensor, axis=(2, 3))

  _, top_matrices = tf.nn.top_k(tf.reshape(submatrices_sum, [-1]),
                                tf.minimum(numer_of_top_submatrices,
                                           tf.size(submatrices_sum)))
  top_row = top_matrices // subm_cols
  top_column = top_matrices % subm_cols
  result = tf.stack([top_row, top_column], axis=-1)

  with tf.Session() as sess:
      return sess.run(result, feed_dict={input: image, submatrix_dims: m,
                                         numer_of_top_submatrices: n})


image_1 = [
          [1, 1, 4, 4],
          [1, 1, 4, 4],
          [3, 3, 2, 2],
          [3, 3, 2, 2],
          ]
image_2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [10, 10, 10, 4, 5, 1, 0, 0, 0, 0],
           [10, 10, 10, 4, 5, 1, 0, 0, 0, 0],
           [5, 3, 3, 4, 5, 5, 8, 8, 8, 1],
           [5, 9, 3, 4, 5, 5, 7, 8, 9, 10],
           [5, 4, 3, 2, 1, 1, 7, 8, 9, 10],
           [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
          ]
n = 1
m = 2
print(get_n_most_salient_regions_of_size_m_by_m(image_2, n, [m, m]))


def get_permutatuins():
  pass
