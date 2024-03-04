from source.dataProcess.util import load_h5file

attr, edge = load_h5file('0000/00000001', '../../data')
print(attr, edge)
