# import torch
#
# from utils.analysis import get_abs_and_relative_positions
#
# tokens = torch.tensor([
#     [0,0,0,1],
#     [0,0,0,1],
#     [0,0,1,1],
#     [0,0,1,1],
#     [0,1,1,1],
#     [0,1,1,1],
#     [1,1,1,1],
#     [1,1,1,1]
# ])
#
# mask = torch.tensor([
#     [1,1,1,1],
#     [0,0,0,0],
#     [1,1,1,1],
#     [0,0,0,0],
#     [1,1,1,1],
#     [0,0,0,0],
#     [1,1,1,1],
#     [0,0,0,0],
# ])
# print(mask.shape)
# print(get_abs_and_relative_positions(mask, tokens, 0, batch_first=True))
from utils.analysis import add_distribution_to_file, prepare_rel_pos_count

info = {20.0: 19921, 80.0: 19786, 40.0: 19673, 60.0: 19514, 10.0: 19416, 50.0: 19283, 90.0: 19179, 30.0: 19086,
     70.0: 19012, 100.0: 11324, 0.0: 8436}

info = prepare_rel_pos_count(info)

add_distribution_to_file(info,"Random 10", "test.txt" )

df = add_distribution_to_file(info,"Random 20", "test.txt" )
#
#
df.plot(x="percentages")
import matplotlib.pyplot as plt
plt.show()
#
