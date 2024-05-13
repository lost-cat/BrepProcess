from dgl.data import load_graphs
import occwl.io

graph, _ = load_graphs('../data/dgl/0000/00000007.bin')

graph = graph[0]
print(graph.ndata['x'])
print(graph.ndata['uv_attrs'])

print(graph.edata['x'])
print(graph.edata['uv_attrs'])
