"""
Verify the implementation by comparing the analytically computed Hessian-diagonal against the MC-estimated one.
"""


from constants import ASSETS
from main import parser, run
from utilities import plot


cmd = "--optimizer adam --widths [32]*1 --trackers exact_hess_diag hess_diag"
save_fn = f"{ASSETS}/sgd/check-hess-estimation.png"
ckpts = [1, 3, 5, 7, 9, 10]

print(f"*** python main.py {cmd} ***")

args = parser.parse_args(cmd.split())
_, out = run(args)
tracked_vals = out[1:]
labels = ("Exact Hess Diag", "Estimated Hess Diag")
log_every = args.epochs // args.ckpts
plot(*tracked_vals, ckpts, *labels, log_every, save_fn=save_fn)