# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
"""Convert NumPy and MPI datatypes."""

from mpi4py import MPI

__all__ = ['fromnumpy', 'tonumpy']


def _get_datatype(dtype):
    # pylint: disable=protected-access
    return MPI._typedict.get(dtype.char)


def _get_typecode(datatype):
    # pylint: disable=protected-access
    return MPI._typecode(datatype)


def fromnumpy(dtype):
    """Convert NumPy datatype to MPI."""
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements

    # structured data type
    if dtype.fields:
        itemsize = dtype.itemsize
        align = dtype.isalignedstruct
        fields = [dtype.fields[name] for name in dtype.names]
        datatypes = []
        blocklengths = [1] * len(fields)
        displacements = [dsp for _, dsp in fields]
        try:
            for ftype, _ in fields:
                datatypes.append(fromnumpy(ftype))
            datatype = MPI.Datatype.Create_struct(
                blocklengths, displacements, datatypes)
        finally:
            for mtp in datatypes:
                mtp.Free()
        if align:
            return datatype
        try:
            return datatype.Create_resized(0, itemsize)
        finally:
            datatype.Free()

    # subarray data type
    if dtype.subdtype:
        base, shape = dtype.subdtype
        datatype = fromnumpy(base)
        try:
            if len(shape) == 1 and shape[0] > 1:
                return datatype.Create_contiguous(shape[0])
            else:
                return datatype.Create_subarray(
                    shape, shape, (0,) * len(shape))
        finally:
            datatype.Free()

    # elementary data type
    datatype = _get_datatype(dtype)
    if datatype is None:
        raise ValueError("cannot convert NumPy datatype to MPI")
    return datatype.Dup()


def tonumpy(datatype):
    """Convert MPI datatype to NumPy."""
    # pylint: disable=too-many-locals
    # pylint: disable=too-many-branches
    # pylint: disable=too-many-statements
    # pylint: disable=too-many-return-statements

    # try:
    #     from numpy import dtype
    # except ImportError:
    #     dtype = lambda arg: arg
    dtype = lambda arg: arg

    _, _, _, combiner = datatype.Get_envelope()

    # predefined datatype
    if combiner == MPI.COMBINER_NAMED:
        typecode = _get_typecode(datatype)
        if typecode is not None:
            return dtype(typecode)
        raise ValueError("cannot convert MPI datatype to NumPy")

    # user-defined datatype
    basetype, _, info = datatype.decode()
    datatypes = [basetype]
    try:
        # duplicated datatype
        if combiner == MPI.COMBINER_DUP:
            return tonumpy(basetype)

        # contiguous datatype
        if combiner == MPI.COMBINER_CONTIGUOUS:
            count = info['count']
            return dtype((tonumpy(basetype), count))

        # vector datatype
        if combiner in (MPI.COMBINER_VECTOR,
                        MPI.COMBINER_HVECTOR):
            npytype = tonumpy(basetype)
            count = info['count']
            blocklength = info['blocklength']
            stride = info['stride']
            if combiner == MPI.COMBINER_VECTOR:
                _, extent = basetype.Get_extent()
                stride *= extent
            names = list(map('f{}'.format, range(count)))
            formats = [(npytype, blocklength)] * count
            offsets = [stride * i for i in range(count)]
            return dtype({'names': names,
                          'formats': formats,
                          'offsets': offsets})

        # indexed datatype
        if combiner in (MPI.COMBINER_INDEXED,
                        MPI.COMBINER_HINDEXED,
                        MPI.COMBINER_INDEXED_BLOCK,
                        MPI.COMBINER_HINDEXED_BLOCK):
            npytype = tonumpy(basetype)
            displacements = info['displacements']
            if combiner in (MPI.COMBINER_INDEXED,
                            MPI.COMBINER_HINDEXED):
                blocklengths = info['blocklengths']
            else:
                blocklengths = [info['blocklength']] * len(displacements)
            stride = 1
            aligned = True
            _, extent = datatype.Get_extent()
            if combiner in (MPI.COMBINER_INDEXED,
                            MPI.COMBINER_INDEXED_BLOCK):
                _, stride = basetype.Get_extent()
            if combiner in (MPI.COMBINER_HINDEXED,
                            MPI.COMBINER_HINDEXED_BLOCK):
                aligned = False
            names = list(map('f{}'.format, range(len(blocklengths))))
            formats = [(npytype, blen) for blen in blocklengths]
            offsets = [disp * stride for disp in displacements]
            return dtype({'names': names,
                          'formats': formats,
                          'offsets': offsets,
                          'itemsize': extent,
                          'aligned': aligned})

        # subarray datatype
        if combiner == MPI.COMBINER_SUBARRAY:
            sizes = info['sizes']
            subsizes = info['subsizes']
            starts = info['starts']
            order = info['order']
            assert subsizes == sizes
            assert min(starts) == max(starts) == 0
            assert order == MPI.ORDER_C
            return dtype((tonumpy(basetype), tuple(sizes)))

        # struct datatype
        aligned = True
        if combiner == MPI.COMBINER_RESIZED:
            if basetype.combiner == MPI.COMBINER_STRUCT:
                assert info['lb'] == 0
                assert info['extent'] == basetype.size
                aligned = False
                combiner = MPI.COMBINER_STRUCT
                _, _, info = basetype.decode()
                datatypes.pop().Free()
        if combiner == MPI.COMBINER_STRUCT:
            _, extent = datatype.Get_extent()
            datatypes = info['datatypes']
            blocklengths = info['blocklengths']
            displacements = info['displacements']
            names = list(map('f{}'.format, range(len(datatypes))))
            formats = list(zip(map(tonumpy, datatypes), blocklengths))
            return dtype({'names': names,
                          'formats': formats,
                          'offsets': displacements,
                          'itemsize': extent,
                          'aligned': aligned})

        # Fortran 90 datatype
        combiner_f90 = (
            MPI.COMBINER_F90_INTEGER,
            MPI.COMBINER_F90_REAL,
            MPI.COMBINER_F90_COMPLEX,
        )
        if combiner in combiner_f90:
            datatypes.pop()
            typesize = datatype.size
            typecode = 'ifc'[combiner_f90.index(combiner)]
            return dtype('{}{:d}'.format(typecode, typesize))

        raise ValueError("cannot convert MPI datatype to NumPy")
    finally:
        for _tp in datatypes:
            if not _tp.is_predefined:
                _tp.Free()