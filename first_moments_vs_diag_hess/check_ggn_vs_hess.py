"""
Verify that the GGN and Hessian diagonals match.
"""


from driver import driver

from constants import ASSETS


cmd = "--optimizer adam --widths [32]*1 --trackers ggn_diag hess_diag"
save_fn = f"{ASSETS}/adam/check-ggn-vs-hess.png"
ckpts = [1, 3, 5, 7, 9, 10]

print(f"*** python main.py {cmd} ***")
driver(cmd, ckpts, save_fn)