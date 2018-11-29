
namespace tick {

template <class T>
class Array {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Array(const T *data, const size_t _size)
      : v_data(data), _size(_size) {}
  Array(Array &&that) : _size(that._size), v_data(that.v_data) {}
  const T *data() const { return v_data; }
  const size_t &size() const { return _size; }
  T dot(const Array<T> &that) const {
    return dot(that.v_data);
  }

  T dot(const T *const that) const {
    T result = 0;
    for (uint64_t i = 0; i < this->_size; i++) result += this->v_data[i] * that[i];
    return result;
  }

  const T &operator[](int i) const { return v_data[i]; }

 private:
  const T *v_data;
  const size_t _size;
  Array() = delete;
  Array(Array &that) = delete;
  Array(const Array &that) = delete;
  Array(const Array &&that) = delete;
  Array &operator=(Array &that) = delete;
  Array &operator=(Array &&that) = delete;
  Array &operator=(const Array &that) = delete;
  Array &operator=(const Array &&that) = delete;
};

template <class T>
class Array2DRaw {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Array2DRaw(const T *_data, const size_t *_info)
      : v_data(_data),
        m_cols(&_info[0]),
        m_rows(&_info[1]),
        m_size(&_info[2]) {}
  Array2DRaw(Array2DRaw &&that)
      : v_data(that.v_data),
        m_cols(that.m_cols),
        m_rows(that.m_rows),
        m_size(that.m_size) {}
  T &operator[](int i) { return v_data[i]; }
  const T *data() const { return v_data; }
  Array<T> row(size_t i) const {
    return Array<T>(&v_data[i * (*m_cols)], *m_cols);
  }
  const T *row_raw(size_t i) const { return &v_data[i * (*m_cols)]; }

  const size_t &cols() const { return *m_cols; }
  const size_t &rows() const { return *m_rows; }
  const size_t &size() const { return *m_size; }

 private:
  const T *v_data;
  const size_t *m_cols, *m_rows, *m_size;

  Array2DRaw() = delete;
  Array2DRaw(Array2DRaw &that) = delete;
  Array2DRaw(const Array2DRaw &that) = delete;
  Array2DRaw(const Array2DRaw &&that) = delete;
  Array2DRaw &operator=(Array2DRaw &that) = delete;
  Array2DRaw &operator=(Array2DRaw &&that) = delete;
  Array2DRaw &operator=(const Array2DRaw &that) = delete;
  Array2DRaw &operator=(const Array2DRaw &&that) = delete;
};

template <class T>
class Sparse {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Sparse(const T *data, const size_t _size, const INDEX_TYPE *indices)
      : v_data(data), _size(_size), indices(indices) {}
  Sparse(Sparse &&that) : _size(that._size) {
    this->v_data = that.v_data;
    this->indices = that.indices;
  };
  const T &operator[](int i) const { return v_data[i]; }
  const T *data() const { return v_data; }
  const size_t &size() const { return _size; }
  T dot(const Sparse<T> &that) const {
    T result = 0;
    uint64_t i1 = 0, i2 = 0;
    while (true) {
      if (i1 >= that._size) break;
      while (i2 < this->_size && this->indices()[i2] < that.indices[i1]) {
        i2++;
      }
      if (i2 >= this->_size) break;
      if (this->indices()[i2] == that.indices[i1]) {
        result += that.v_data[i2] * this->v_data[i1++];
      } else {
        while (i1 < that._size && this->indices()[i2] > that.indices[i1]) {
          i1++;
        }
      }
    }
    return result;
  }
  T dot(const T *const that) const {
    T result = 0;
    for (uint64_t i = 0; i < this->_size; i++) result += this->v_data[i] * that[this->indices[i]];
    return result;
  }

 private:
  const T *v_data;
  const size_t _size;
  const INDEX_TYPE *indices;
  Sparse() = delete;
  Sparse(Sparse &that) = delete;
  Sparse(const Sparse &that) = delete;
  Sparse(const Sparse &&that) = delete;
  Sparse &operator=(Sparse &that) = delete;
  Sparse &operator=(Sparse &&that) = delete;
  Sparse &operator=(const Sparse &that) = delete;
  Sparse &operator=(const Sparse &&that) = delete;
};

template <class T>
class Sparse2DRaw {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Sparse2DRaw(const T *_data, const size_t *_info, const INDEX_TYPE *_indices,
              const INDEX_TYPE *_row_indices)
      : v_data(_data),
        m_cols(&_info[0]),
        m_rows(&_info[1]),
        m_size(&_info[2]),
        v_indices(_indices),
        v_row_indices(_row_indices) {}

  Sparse2DRaw(Sparse2DRaw &&that)
      : v_data(that.v_data),
        m_cols(that.m_cols),
        m_rows(that.m_rows),
        m_size(that.m_size),
        v_indices(that.v_indices),
        v_row_indices(that.v_row_indices) {}
  T &operator[](int i) { return v_data[i]; }
  Sparse<T> row(size_t i) const {
    return Sparse<T>(v_data + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                     v_indices + v_row_indices[i]);
  }
  const T *data() const { return v_data; }
  const T *row_raw(size_t i) const { return v_data + v_row_indices[i]; }
  INDEX_TYPE row_size(size_t i) const { return v_row_indices[i + 1] - v_row_indices[i]; }
  const INDEX_TYPE *indices() const { return v_indices; }
  const INDEX_TYPE *row_indices(size_t i) const { return v_indices + v_row_indices[i]; }
  const INDEX_TYPE *row_indices() const { return v_row_indices; }

  const size_t &cols() const { return *m_cols; }
  const size_t &rows() const { return *m_rows; }
  const size_t &size() const { return *m_size; }

 private:
  const T *v_data;
  const size_t *m_cols, *m_rows, *m_size;
  const INDEX_TYPE *v_indices, *v_row_indices;

  Sparse2DRaw() = delete;
  Sparse2DRaw(Sparse2DRaw &that) = delete;
  Sparse2DRaw(const Sparse2DRaw &that) = delete;
  Sparse2DRaw(const Sparse2DRaw &&that) = delete;
  Sparse2DRaw &operator=(Sparse2DRaw &that) = delete;
  Sparse2DRaw &operator=(Sparse2DRaw &&that) = delete;
  Sparse2DRaw &operator=(const Sparse2DRaw &that) = delete;
  Sparse2DRaw &operator=(const Sparse2DRaw &&that) = delete;
};

template <class T>
class Sparse2DList {
 private:
  using INDEX_TYPE = INDICE_TYPE;
  static constexpr size_t INFO_SIZE = 5;

 public:
  Sparse2DList(std::vector<T> &data, std::vector<size_t> &info, std::vector<INDEX_TYPE> &_indices,
               std::vector<INDEX_TYPE> &_rows_indices)
      : v_data(data.data()),
        v_info(info.data()),
        v_indices(_indices.data()),
        v_row_indices(_rows_indices.data()) {}

  Sparse2DRaw<T> operator[](size_t i) const {
    return Sparse2DRaw<T>(v_data + (v_info[(i * INFO_SIZE) + 3]), &v_info[i * INFO_SIZE],
                          v_indices + (v_info[(i * INFO_SIZE) + 3]),
                          v_row_indices + (v_info[(i * INFO_SIZE) + 4]));
  }

  const INDEX_TYPE *indices() const { return v_indices; }
  const INDEX_TYPE *row_indices() const { return v_row_indices; }
  const size_t *info() const { return v_info; }

 private:
  T *v_data;
  size_t *v_info;
  INDEX_TYPE *v_indices, *v_row_indices;

  Sparse2DList() = delete;
  Sparse2DList(Sparse2DList &that) = delete;
  Sparse2DList(const Sparse2DList &that) = delete;
  Sparse2DList(Sparse2DList &&that) = delete;
  Sparse2DList(const Sparse2DList &&that) = delete;
  Sparse2DList &operator=(Sparse2DList &that) = delete;
  Sparse2DList &operator=(Sparse2DList &&that) = delete;
  Sparse2DList &operator=(const Sparse2DList &that) = delete;
  Sparse2DList &operator=(const Sparse2DList &&that) = delete;
};

}  // namespace tick
