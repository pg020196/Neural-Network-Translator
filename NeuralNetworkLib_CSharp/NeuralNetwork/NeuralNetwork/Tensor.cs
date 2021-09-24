using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;

namespace Tensor
{
    public static class IntExtension
    {
        public static string arrayToString(this int[] arr)
        {
            if (arr.Length < 1)
                return "()";

            StringBuilder sb = new StringBuilder();
            sb.Append("(");
            sb.Append(arr[0]);
            for (int i = 1; i < arr.Length; i++)
                sb.Append(", " + arr[i]);
            sb.Append(")");
            return sb.ToString();
        }
    }


    public class Tensor<T>
    {
        /*******************************************************************************\
         *  PRIVATE FIELDS
        \*******************************************************************************/

        // Helper to carry out calculations depending on T
        protected static readonly ICalculator<T> calculator = Calculators.GetInstance<T>();
        // Total number of elements in Tensor, i.e. product of all elements in Shape
        public int NumElems { get; }
        // Number of dimensions of tensor
        public int Dims { get; private set; }
        // Shape of tensor
        public int[] Shape { get; private set; }
        // Underlying Data
        private T[] Data { get; set; }

        //private IDataAccessBehavior dataAccessBehavior;

        /*******************************************************************************\
        *  CREATION
        \*******************************************************************************/

        // Constructor for initializing by single element (i.e. 0D array)
        public Tensor(T value)
        {
            Dims = 0;
            Shape = new int[0];
            NumElems = 1;
            Data = new T[] { value };
        }

        // Constructor for initializing by 1D array
        public Tensor(T[] initializer)
        {
            Dims = 1;
            Shape = new int[1] { initializer.Length };
            NumElems = initializer.Length;
            Data = initializer;
        }

        // Constructor for creating a Tensor object with given Shape
        // and initializing all values with zero.
        // Set to private due to potential confusion with 
        // Tensor(params T[]) for extern users.
        // Extern users should use static method
        // Tensor<T>.zeros(int[] Shape) for same effect
        protected Tensor(params int[] shape)
        {
            if (shape.Length == 0) // 0D array
            {
                Dims = 0;
                Shape = new int[0];
                NumElems = 1;
                Data = new T[1];
            }
            else
            {
                for (int i = 0; i < shape.Length; i++)
                    if (shape[i] < 0)
                        throw new ArgumentException("Each dimension must be non-negative");

                // initialize private fields
                Dims = shape.Length;
                //this.Shape = Shape;
                Shape = shape;

                // compute total number of elements as product of Shape components
                NumElems = shape.Aggregate(1, (a, b) => a * b);

                // create Data array to hold tensor elements
                Data = new T[NumElems];
            }
        }

        // Constructor for creating Tensor object from existing Tensor object.
        // Values are set as in the given object, and depending on the argument copy
        // underlying Data array is referenced or copied.
        public Tensor(Tensor<T> otherTensor, bool copy = false)
        {
            NumElems = otherTensor.NumElems;
            Dims = otherTensor.Dims;
            Shape = otherTensor.Shape;

            if (!copy) // reference the underlying Data array
                Data = otherTensor.Data;
            else // copy the Data array
            {
                Data = new T[otherTensor.NumElems];
                for (int i = 0; i < otherTensor.NumElems; i++)
                    Data[i] = otherTensor.Data[i];
            }
        }


        // Create Tensor with given Shape and initialize values with zero
        public static Tensor<T> zeros(params int[] shape)
        {
            return new Tensor<T>(shape);
        }


        // Create Tensor object with given Shape and initialize values with one
        public static Tensor<T> ones(params int[] shape)
        {
            Tensor<T> t = new Tensor<T>(shape);
            for (int i = 0; i < t.NumElems; i++)
                t.Data[i] = calculator.getOne();
            return t;
        }


        public static Tensor<T> arange(params T[] bounds)
        {
            T start = calculator.getZero();
            T stop;
            T step = calculator.getOne();

            switch (bounds.Length)
            {
                case 0:
                    throw new ArgumentOutOfRangeException("At least 1 argument must be provided to arange");
                case 1:
                    stop = bounds[0];
                    break;
                case 2:
                    start = bounds[0];
                    stop = bounds[1];
                    break;
                case 3:
                    start = bounds[0];
                    stop = bounds[1];
                    step = bounds[2];
                    break;
                default:
                    throw new ArgumentOutOfRangeException("At most 3 arguments may be provided to arange");
            }

            T range = calculator.subtract(stop, start);
            int numElements = calculator.ceilInt(calculator.divide(range, step));
            if (numElements <= 0)
                return new Tensor<T>(new int[] { 0 });

            var result = new Tensor<T>(numElements);

            result[0] = start;
            for (int i = 1; i < numElements; i++)
                result[i] = calculator.add(result[i - 1], step);

            return result;
        }

        public static Tensor<T> linspace(T start, T stop, int num=100)
        {
            //if (!calculator.isGreater(stop, start))
            //    throw new ArgumentException("stop must be greater than start");

            var result = new Tensor<T>(num);

            T increment = calculator.divide(calculator.subtract(stop, start), (num-1));
            result[0] = start;
            for (int i = 1; i < num; i++)
                result[i] = calculator.add(result[i - 1], increment);

            return result;
        }

        // Create Tensor object with given Shape and initialize values
        // with uniformly distributed random values
        public static Tensor<T> randUniform(double min, double max, params int[] shape)
        {
            Tensor<T> t = new Tensor<T>(shape);
            for (int i = 0; i < t.NumElems; i++)
                t.Data[i] = calculator.getRandUniform(min, max);
            return t;
        }


        // Create Tensor object with given Shape and initialize values
        // with normally distributed random values
        public static Tensor<T> randNormal(double mean, double std, params int[] shape)
        {
            Tensor<T> t = new Tensor<T>(shape);
            for (int i = 0; i < t.NumElems; i++)
                t.Data[i] = calculator.getRandNormal(mean, std);
            return t;
        }


        /*******************************************************************************\
         *  INDEXING
        \*******************************************************************************/

        // Convert index list to 1D index
        private int getIndex(List<int> indexList)
        {
            if (Dims == 0) // Special handling of scalar Tensor objects
            {
                if (indexList.Count == 1 && indexList[0] == 0)
                    return 0;
                else
                    throw new ArgumentOutOfRangeException();
            }

            // check that as many indices are provided as there are dimensions
            if (indexList.Count != Dims)
                throw new ArgumentException("Argument indexList must be of length " +
                    $"{Dims}, but is of length {indexList.Count}");

            // check bounds
            for (int i = 0; i < Dims; i++)
            {
                if (indexList[i] < 0 || indexList[i] >= Shape[i])
                    throw new ArgumentException("All indexers must be within {{0," +
                        $" dims[i]-1}}, but indexList[{i}] is {indexList[i]}");
            }

            // compute index
            int index = indexList[0];
            for (int i = 1; i < indexList.Count; i++)
            {
                index *= Shape[i];
                index += indexList[i];
            }
            return index;
        }

        // Convert 1D index to index list
        private List<int> getIndexList(int index)
        {
            // TODO: refactor as iterator
            // check bounds
            if (index < 0 || index >= NumElems)
                throw new ArgumentException("Index must be between 0 and the number" +
                    $" of elements in the tensor object ({NumElems}), but is {index}");

            // List<int> indexList = new List<int>(Dims);
            int[] indexList = new int[Dims];

            // indexList[Dims-1] = index % Shape[Dims - 1];

            // Determine each dimension by number of blocks that fit in subtensor
            int blockSize = 1;
            for (int i = Dims - 1; i >= 0; i--)
            {
                indexList[i] = (index / blockSize) % Shape[i];
                blockSize *= Shape[i];
            }
            return indexList.ToList();
        }

        // Overload the indexing operator
        public T this[List<int> indexList]
        {
            get
            {
                return Data[getIndex(indexList)];
            }
            set
            {
                Data[getIndex(indexList)] = value;
            }
        }

        // Overload the indexing operator
        public T this[params int[] indexList]
        {
            get
            {
                return Data[getIndex(indexList.ToList())];
            }
            set
            {
                Data[getIndex(indexList.ToList())] = value;
            }
        }

        public Tensor<T> this[params Range[] ranges]
        {
            get
            {
                return getSlices(ranges);
            }
        }

        // this.getSlice(i, a..b) --> numpy equivalent: this[:,:,...,a:b,...,:],
        // where a:b is the i-th index (0-based).
        // I.e., the slice is taken from the i-th dimension, while the
        // remaining dimensions are taken completely.
        public Tensor<T> getSlice(int axis, Range range, bool keepdims = false)
        {
            if (axis < -Dims || axis > Dims - 1)
                throw new ArgumentOutOfRangeException("Cannot take slice over axis " +
                    $"{axis} of Tensor with {Dims} dimensions (shape {Shape})");

            // Check if axis is negative, i.e. counted from end, if so transform
            int _axis;
            if (axis < 0)
                _axis = Dims + axis;
            else
                _axis = axis;

            // Check if indices are counted from end, if so transform.
            // Note that by definition of Range, indices counted from end
            // are still positive, in contrast to our definition of axis.
            int r0 = range.Start.Value;
            if (range.Start.IsFromEnd)
                r0 = Shape[_axis] - r0;
            int r1 = range.End.Value;
            if (range.End.IsFromEnd)
                r1 = Shape[_axis] - r1;

            // Handle invalid range
            if (r0 < 0 || r1 > Shape[_axis])
                throw new ArgumentOutOfRangeException($"Slice indices ({r0}:{r1}) out " +
                    $"of bounds of axis {_axis} with dimension {Shape[_axis]}");

            // Handle case r0 > r1
            if (r0 >= r1) // empty slice
            {
                // Determine resulting Shape
                var resShape = Shape.Select((s, i) => (_axis == i) ? 0 : s).ToArray();
                return new Tensor<T>(resShape);
            }
            else if (r1 - r0 == 1)
            {
                return singleAxisSingleElementSlicing(_axis, r0, keepdims);
            }
            else if (r1 - r0 == Shape[_axis])
            {
                // slice equals full array
                return this;
            }
            else
            {
                return singleAxisMultiElementSlicing(_axis, r0, r1);
            }
        }

        private Tensor<T> singleAxisSingleElementSlicing(int _axis, int r0, bool keepdims)
        {
            int[] resShape;
            // Determine resulting Shape
            if (keepdims)
                resShape = Shape.Select((s, i) => (_axis == i) ? 1 : s).ToArray();
            else
                resShape = Shape.Where((s, i) => i != _axis).ToArray();

            List<int> indexList;
            Tensor<T> res = new Tensor<T>(resShape);
            for (int i = 0; i < res.NumElems; i++)
            {
                indexList = res.getIndexList(i);
                if (keepdims)
                    indexList[_axis] = r0;
                else
                    indexList.Insert(_axis, r0);
                res.Data[i] = this[indexList];
            }
            return res;
        }

        private Tensor<T> singleAxisMultiElementSlicing(int _axis, int r0, int r1)
        {
            // Determine resulting Shape
            int[] resShape = Shape.Select((s, i) => (_axis == i) ? r1 - r0 : s).ToArray();

            List<int> indexList;
            Tensor<T> res = new Tensor<T>(resShape);
            for (int i = 0; i < res.NumElems; i++)
            {
                indexList = res.getIndexList(i);
                indexList[_axis] += r0;
                res.Data[i] = this[indexList];
            }
            return res;
        }

        public Tensor<T> getSlices(params Range[] ranges)
        {
            if (ranges.Length > Dims)
                throw new ArgumentException($"Too many indices ({ranges.Length}) " +
                    $"for Tensor of shape {this.Shape}");

            Range[] augmentedRanges;
            if (ranges.Length < Dims)
            {
                List<Range> augmentedRangesList = ranges.ToList();
                for (int i = ranges.Length; i < Dims; i++)
                    augmentedRangesList.Add(Range.All);
                augmentedRanges = augmentedRangesList.ToArray();
            }
            else
                augmentedRanges = ranges;

            Tensor<T> res = this;
            for (int axis = Dims - 1; axis >= 0; axis--)
            {
                res = res.getSlice(axis, augmentedRanges[axis]);
            }

            return res;
        }

        public Tensor<T> reshape(params int[] newShape)
        {
            if (NumElems != newShape.Aggregate((a, b) => a * b))
                throw new ArgumentException($"Cannot reshape tensor of shape {tupleToString(Shape)} to shape {tupleToString(newShape)} (mismatching number of elements)");

            // create new tensor object (referencing the same underlying Data array)
            Tensor<T> newT = new Tensor<T>(this)
            {
                Shape = newShape,
                Dims = newShape.Length
            };
            return newT;
        }

        public Tensor<T> flatten()
        {
            return reshape(NumElems);
        }


        /*******************************************************************************\
         *  MATH AND MANIPULATION
        \*******************************************************************************/

        private enum ReduceType
        {
            min,
            mean,
            sum,
            max
        }

        private Tensor<T> reduce(ReduceType reduceType, params int[] axes)
        {
            //if (!(reduceType == ReduceType.min || reduceType == ReduceType.mean || reduceType == ReduceType.max))
            //    throw new ArgumentException($"Unknown parameter type {reduceType}, must be " +
            //        $"mean, min or max");

            if (axes.Length == 0)
            {
                return reduceFlatten(reduceType);
            }
            else if (axes.Length == 1)
            {
                return reduce1D(reduceType, axes[0]);
            }
            else
            {
                return reduceND(reduceType, axes);
            }
        }

        private Tensor<T> reduceFlatten(ReduceType reduceType)
        {
            if (Dims == 0)
            {
                return this;
            }

            switch (reduceType)
            {
                case ReduceType.min:
                    T min = calculator.getPosInfinity();
                    for (int i = 0; i < NumElems; i++)
                        if (calculator.isLess(Data[i], min))
                            min = Data[i];
                    return new Tensor<T>(min);
                case ReduceType.mean:
                    T mean = calculator.getZero();
                    for (int i = 0; i < NumElems; i++)
                        mean = calculator.add(mean, Data[i]);
                    mean = calculator.divide(mean, NumElems);
                    return new Tensor<T>(mean);
                case ReduceType.sum:
                    T sum = calculator.getZero();
                    for (int i = 0; i < NumElems; i++)
                        sum = calculator.add(sum, Data[i]);
                    return new Tensor<T>(sum);
                case ReduceType.max:
                    T max = calculator.getNegInfinity();
                    for (int i = 0; i < NumElems; i++)
                        if (calculator.isGreater(Data[i], max))
                            max = Data[i];
                    return new Tensor<T>(max);
            }
            throw new ArgumentException($"ReduceType {reduceType} is invalid");
        }

        private Tensor<T> reduce1D(ReduceType reduceType, int axis)
        {
            if (axis >= Dims || axis < -Dims)
            {
                throw new ArgumentOutOfRangeException($"Cannot reduce over axis {axis}." +
                    $"Argument must be between {-Dims} and {Dims - 1}");
            }
                
            // work on copy of axis to avoid overriding ref argument
            int _axis = axis;
            // reformat negative axis arguments to positive counterpart
            if (axis < 0)
            {
                _axis = Dims + axis;
            }

            // determine remaining axis and Shape
            var axesRemaining = new List<int>();
            var shapeRemaining = new List<int>();
            for (int i = 0; i < Dims; i++)
            {
                if (_axis != i)
                {
                    axesRemaining.Add(i);
                    shapeRemaining.Add(Shape[i]);
                }
            }

            if (reduceType == ReduceType.min)
            {
                return min1D(_axis, shapeRemaining);
            }
            else if (reduceType == ReduceType.mean)
            {
                return mean1D(_axis, shapeRemaining);
            }
            else if (reduceType == ReduceType.sum)
            {
                return sum1D(_axis, shapeRemaining);
            }
            else // max
            {
                return max1D(_axis, shapeRemaining);
            }
        }

        private Tensor<T> min1D(int axis, List<int> shapeRemaining)
        {
            Tensor<T> reducedTensor = new Tensor<T>(shapeRemaining.ToArray());
            for (int i = 0; i < reducedTensor.NumElems; i++)
            {
                var indexList = reducedTensor.getIndexList(i);
                indexList.Insert(axis, 0);
                T currentMin = calculator.getPosInfinity();
                for (int j = 0; j < Shape[axis]; j++)
                {
                    indexList[axis] = j;
                    if (calculator.isLess(this[indexList], currentMin))
                    {
                        currentMin = this[indexList];
                    }
                }
                reducedTensor.Data[i] = currentMin;
            }
            return reducedTensor;
        }

        private Tensor<T> mean1D(int axis, List<int> shapeRemaining)
        {
            Tensor<T> reducedTensor = new Tensor<T>(shapeRemaining.ToArray());
            for (int i = 0; i < reducedTensor.NumElems; i++)
            {
                var indexList = reducedTensor.getIndexList(i);
                indexList.Insert(axis, 0);
                T aggregator = calculator.getZero();
                for (int j = 0; j < Shape[axis]; j++)
                {
                    indexList[axis] = j;
                    aggregator = calculator.add(aggregator, this[indexList]);
                }
                aggregator = calculator.divide(aggregator, Shape[axis]);
                //indexList.RemoveAt(_axis);
                //reducedTensor[indexList] = aggregator;
                reducedTensor.Data[i] = aggregator;
            }
            return reducedTensor;
        }

        private Tensor<T> sum1D(int axis, List<int> shapeRemaining)
        {
            Tensor<T> reducedTensor = new Tensor<T>(shapeRemaining.ToArray());
            for (int i=0; i < reducedTensor.NumElems; i++)
            {
                var indexList = reducedTensor.getIndexList(i);
                indexList.Insert(axis, 0);
                T sum = calculator.getZero();
                for (int j = 0; j < Shape[axis]; j++)
                {
                    indexList[axis] = j;
                    sum = calculator.add(sum, this[indexList]);
                }
                reducedTensor.Data[i] = sum;                    
            }
            return reducedTensor;
        }

        private Tensor<T> max1D(int axis, List<int> shapeRemaining)
        {
            Tensor<T> reducedTensor = new Tensor<T>(shapeRemaining.ToArray());
            for (int i = 0; i < reducedTensor.NumElems; i++)
            {
                var indexList = reducedTensor.getIndexList(i);
                indexList.Insert(axis, 0);
                T currentMax = calculator.getNegInfinity();
                for (int j = 0; j < Shape[axis]; j++)
                {
                    indexList[axis] = j;
                    if (calculator.isGreater(this[indexList], currentMax))
                        currentMax = this[indexList];
                }
                //indexList.RemoveAt(_axis);
                //reducedTensor[indexList] = currentMax;
                reducedTensor.Data[i] = currentMax;
            }
            return reducedTensor;
        }

        private Tensor<T> reduceND(ReduceType reduceType, params int[] axes)
        {
            // check arguments
            for (int i = 0; i < axes.Length; i++)
                if (axes[i] > Dims || axes[i] < -Dims)
                    throw new ArgumentOutOfRangeException($"Cannot reduce over axis {axes[i]}. " +
                        $"Argument must be between {-Dims} and {Dims - 1}");

            // work on copy of axis to avoid overriding ref argument
            int[] _axes = new int[axes.Length];
            axes.CopyTo(_axes, 0);

            // To allow negative axis (i.e. counting from -1 == last dimension)
            // replace negative with positive dimensions
            for (int i = 0; i < _axes.Length; i++)
                if (_axes[i] < 0)
                    _axes[i] = Dims + _axes[i];

            // remove duplicates and sort
            int[] axesCleaned = _axes.Distinct().ToArray();
            Array.Sort(axesCleaned);

            // iteratively apply reduction over all axis
            Tensor<T> res = new Tensor<T>(this);
            for (int i = axesCleaned.Length - 1; i >= 0; i--)
                res = res.reduce1D(reduceType, axesCleaned[i]);

            return res;
        }

        // Non-static / member version of min
        public Tensor<T> min(params int[] axes)
        {
            return reduce(ReduceType.min, axes);
        }

        // Non-static / member version of mean
        public Tensor<T> mean(params int[] axes)
        {
            return reduce(ReduceType.mean, axes);
        }

        public Tensor<T> sum(params int[] axes)
        {
            return reduce(ReduceType.sum, axes);
        }

        // Non-static / member version of max
        public Tensor<T> max(params int[] axes)
        {
            return reduce(ReduceType.max, axes);
        }

        // A function to compute the dot product of two tensors
        // as defined in https://numpy.org/doc/stable/reference/generated/numpy.dot.html
        public Tensor<T> dot(Tensor<T> t1)
        {
            if (Dims == 0)
            {
                return t1.scalarMultiply(this[0]);
            }

            if (t1.Dims == 0)
            {
                return scalarMultiply(t1[0]);
            }

            Tensor<T> t0_;


            t0_ = prepareLeft(out int[] t0LeadingShape);

            if (t1.Dims <= 2)
            {
                return t0_.dotProductRightDimsLessEqual2(t1, t0LeadingShape);
            }
            else // t1.Dims > 2
            {
                return t0_.dotProductRightDimsGreater2(t1, t0LeadingShape);
            }
        }

        public Tensor<T> dot(T scalar)
        {
            return dot(new Tensor<T>(scalar));
        }

        private Tensor<T> prepareLeft(out int[] t0LeadingShape)
        {
            Tensor<T> t0_;
            if (Dims == 1)
            {
                t0LeadingShape = new int[] { };
                t0_ = reshape(1, Shape[0]);
            }
            else if (Dims > 2)
            {
                t0LeadingShape = Shape[0..^1];
                int t0FlattenedLengthLeading = t0LeadingShape.Aggregate((a, b) => a * b);
                t0_ = reshape(t0FlattenedLengthLeading, Shape[Dims - 1]);
            }
            else // Dims == 2
            {
                t0LeadingShape = Shape[0..1];
                t0_ = this;
            }
            return t0_;
        }

        private Tensor<T> dotProductRightDimsLessEqual2(Tensor<T> t1, int[] t0LeadingShape)
        {
            Tensor<T> t1_;
            int[] t1TrailingShape;

            if (t1.Dims == 1)
            {
                t1TrailingShape = new int[] { };
                t1_ = t1.reshape(t1.Shape[0], 1);
            }
            else if (t1.Dims > 2)
            {
                throw new NotImplementedException();
            }
            else // t1.Dims == 2
            {
                t1TrailingShape = t1.Shape[1..2];
                t1_ = t1;
            }

            var result = matrixMultiply(t1_);
            var resultShape = new List<int>();
            if (t0LeadingShape.Length == 0 && t1TrailingShape.Length == 0)
            {
                return new Tensor<T>(result[0, 0]);
            }
            else
            {
                resultShape.AddRange(t0LeadingShape);
                resultShape.AddRange(t1TrailingShape);
                return result.reshape(resultShape.ToArray());
            }
            
        }

        private Tensor<T> dotProductRightDimsGreater2(Tensor<T> t1, int[] t0LeadingShape)
        {
            Tensor<T> t1_;
            int[] t1TrailingShape;

            int[] t1LeadingShape = t1.Shape[0..^2];
            int t1FlattenedLengthLeading = t1LeadingShape.Aggregate((a, b) => a * b);

            int t1ProductAxisLength = t1.Shape[t1.Dims - 2];

            t1TrailingShape = t1.Shape[^1..^0];
            int t1FlattenedLengthTrailing = t1TrailingShape.Aggregate((a, b) => a * b);

            int[] t1TmpShape = { t1FlattenedLengthLeading, t1ProductAxisLength, t1FlattenedLengthTrailing };
            t1_ = t1.reshape(t1TmpShape);

            int[] resultTmpShape = { Shape[0], t1_.Shape[0], t1_.Shape[2] };
            var result = zeros(resultTmpShape);

            for (int i = 0; i < resultTmpShape[0]; i++)
            {
                for (int j = 0; j < resultTmpShape[1]; j++)
                {
                    for (int k = 0; k < resultTmpShape[2]; k++)
                    {
                        result[i, j, k] = calculator.getZero();
                        for (int l = 0; l < t1ProductAxisLength; l++)
                        {
                            result[i, j, k] = calculator.add(result[i, j, k], calculator.multiply(this[i, l], t1_[j, l, k]));
                        }
                    }
                }
            }

            var resultShape = new List<int>();
            resultShape.AddRange(t0LeadingShape);
            resultShape.AddRange(t1LeadingShape);
            resultShape.Add(t1FlattenedLengthTrailing);
            return result.reshape(resultShape.ToArray());
        }

        public Tensor<T> scalarMultiply(T scalar, bool inplace=false)
        {
            Tensor<T> resultTensor;
            if (inplace)
                resultTensor = this;
            else
                resultTensor = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
            {
                resultTensor.Data[i] = calculator.multiply(resultTensor.Data[i], scalar);
            }
            return resultTensor;
        }

        public Tensor<T> scalarMultiply(double scalar, bool inplace=false)
        {
            T value = calculator.getValue(scalar);
            return scalarMultiply(value, inplace);
        }

        // Matrix multiplication of two rank-2 Tensors
        public Tensor<T> matrixMultiply(Tensor<T> t1)
        {
            if (Shape[1] != t1.Shape[0])
                throw new ArgumentException($"Shapes {tupleToString(Shape)} and {tupleToString(t1.Shape)} are incompatible for matrix product");

            Tensor<T> res = new Tensor<T>(Shape[0], t1.Shape[1]);
            for (int r0 = 0; r0 < Shape[0]; r0++)
            {
                for (int c1 = 0; c1 < t1.Shape[1]; c1++)
                {
                    var res_rc = calculator.getZero();
                    for (int i = 0; i < Shape[1]; i++)
                    {
                        res_rc = calculator.add(res_rc, calculator.multiply(this[r0, i], t1[i, c1]));
                    }
                    res[r0, c1] = res_rc;
                }
            }
            return res;
        }

        public Tensor<T> add(Tensor<T> t1, bool inplace = false)
        {
            for (int i = 0; i < Dims; i++)
                if (Shape[i] != t1.Shape[i])
                    throw new ArgumentException("Cannot elementwise add two tensors of different shape");

            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.add(Data[i], t1.Data[i]);
            return result;
        }

        public Tensor<T> subtract(Tensor<T> t1, bool inplace = false)
        {
            for (int i = 0; i < Dims; i++)
                if (Shape[i] != t1.Shape[i])
                    throw new ArgumentException("Cannot elementwise subtract two tensors of different shape");

            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.subtract(Data[i], t1.Data[i]);
            return result;
        }

        public Tensor<T> multiply(Tensor<T> t1, bool inplace=false)
        {
            for (int i = 0; i < Dims; i++)
                if (Shape[i] != t1.Shape[i])
                    throw new ArgumentException("Cannot elementwise multiply two tensors of different shape");

            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.multiply(Data[i], t1.Data[i]);
            return result;
        }

        public Tensor<T> divide(Tensor<T> t1, bool inplace = false)
        {
            for (int i = 0; i < Dims; i++)
                if (Shape[i] != t1.Shape[i])
                    throw new ArgumentException("Cannot elementwise divide two tensors of different shape");

            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.divide(Data[i], t1.Data[i]);
            return result;
        }

        public Tensor<T> exp(bool inplace=false)
        {
            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.exp(result.Data[i]);
            return result;
        }

        public Tensor<T> negExp(bool inplace=false)
        {
            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.exp(calculator.negate(result.Data[i]));
            return result;
        }

        public Tensor<T> negate(bool inplace = false)
        {
            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.negate(result.Data[i]);
            return result;
        }

        public Tensor<T> invert(bool inplace=false)
        {
            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.invert(result.Data[i]);
            return result;
        }

        public Tensor<T> abs(bool inplace=false)
        {
            Tensor<T> result;
            if (inplace)
                result = this;
            else
                result = new Tensor<T>(this, copy: true);
            for (int i = 0; i < NumElems; i++)
                result.Data[i] = calculator.abs(result.Data[i]);
            return result;
        }

        /*******************************************************************************\
         *  OUTPUT FORMATTING
        \*******************************************************************************/

        public static string tupleToString(int[] tuple)
        {
            if (tuple.Length == 0)
                return "( )";
            else if (tuple.Length == 1)
                return $"({tuple[0]},)";
            else
                return "(" + tuple.Select(x => string.Format("{0}", x))
                    .Aggregate((current, next) => current + ", " + next) + ")";
        }


        // call the ToString(string, string, bool) method with default arguments
        public override string ToString()
        {
            return ToString();
        }

        public string ToString(string culture = "en-US", bool display_brackets = true, int indent = 4)
        {
            // Set "culture" to US for using '.' as decimal point
            IFormatProvider provider = CultureInfo.CreateSpecificCulture(culture);
            //string format = "{0,11:g5}";

            //T max = this.max();
            //T min = this.min();

            // TODO
            string format = "{0,9:###0.0000}";

            StringBuilder sb = new StringBuilder();

            if (Dims == 0)
            {
                sb.Append($"Tensor(Shape: {tupleToString(Shape)}, DType: {typeof(T)}, Data:\n");
                if (indent > 0)
                    sb.Append(new String(' ', indent));
                sb.Append(String.Format(provider, format, Data[0]));
                sb.Append("\n)");
                return sb.ToString();
            }
            else
            {
                int numDivisibleDims = 0;
                int blockSize = Shape[Dims - 1];
                int d = Dims - 1;

                // Formatting beginning of tensor
                sb.Append($"Tensor(Shape: {tupleToString(Shape)}, DType: {typeof(T)}, Data:\n");
                if (indent > 0)
                    sb.Append(new String(' ', indent));
                if (display_brackets)
                    sb.Append(new String('[', Dims));
                sb.Append(String.Format(provider, format, Data[0]));
                if (Shape[Dims - 1] != 1)
                    sb.Append(", ");
                for (int i = 1; i < NumElems; i++)
                {
                    // Block formatting: Determine whether current index is beginning of a
                    // new block. For new block, include line breaks and brackets depending
                    // on block size, determined by
                    // if i is divisible by Shape[Dims-1]
                    //     --> new smallest block
                    // if i is divisible by Shape[Dims-2] * Shape[Dims-1]
                    //     --> new second smallest block
                    // etc.
                    while (d > 0)
                    {
                        if (i % blockSize == 0)
                        {
                            numDivisibleDims++;
                            blockSize *= Shape[d - 1];
                            d--;
                        }
                        else
                            break;
                    }

                    if (numDivisibleDims > 0) // Formatting for new block
                    {
                        if (display_brackets)
                        {
                            sb.Append(new String(']', numDivisibleDims));
                            sb.Append(",");
                            sb.Append(new String('\n', numDivisibleDims));
                            if (indent > 0)
                                sb.Append(new String(' ', indent));
                            sb.Append(new String(' ', Dims - numDivisibleDims));
                            sb.Append(new String('[', numDivisibleDims));
                            sb.Append(String.Format(provider, format, Data[i]));
                            if ((i + 1) % Shape[Dims - 1] != 0)
                                sb.Append(", ");
                        }
                        else
                        {
                            sb.Append(",");
                            sb.Append(new String('\n', numDivisibleDims));
                            if (indent > 0)
                                sb.Append(new String(' ', indent));
                            sb.Append(String.Format(provider, format, Data[i]));
                            if ((i + 1) % Shape[Dims - 1] != 0)
                                sb.Append(", ");
                        }
                        numDivisibleDims = 0;
                        blockSize = Shape[Dims - 1];
                        d = Dims - 1;
                    }
                    else // Regular formatting
                    {
                        sb.Append(String.Format(provider, format, Data[i]));
                        if ((i + 1) % Shape[Dims - 1] != 0)
                            sb.Append(", ");
                    }
                }
                // Formatting end of tensor
                if (display_brackets)
                    sb.Append(new String(']', Dims));
                sb.Append("\n)");
                return sb.ToString();
            }
        }

        /*******************************************************************************\
         *  EQUALITY CHECK
        \*******************************************************************************/

        public override bool Equals(object obj)
        {
            return Equals(obj, 1e-6);
        }

        private bool Equals(object obj, double eps)
        {
            var other = obj as Tensor<T>;

            if (other == null)
            {
                return false;
            }

            if (other.Dims != Dims)
            {
                return false;
            }

            for (var i = 0; i < Shape.Length; i++)
            {
                if (other.Shape[i] != Shape[i])
                {
                    return false;
                }
            }

            T eps_ = calculator.getValue(eps);

            T absDiff;
            for (var i = 0; i < NumElems; i++)
            {
                absDiff = calculator.abs(calculator.subtract(other.Data[i], Data[i]));
                if (calculator.isGreater(absDiff, eps_))
                {
                    return false;
                }
            }

            return true;
        }
    }
}