from driver import driver

from constants import ASSETS


cmd = "--optimizer adam --beta1 0.0 --lr 1e-1 --eps 100"
save_fn = f"{ASSETS}/sgd/expavgsq.png"
ckpts = [1, 3, 5, 7, 9, 10]

print(f"*** python main.py {cmd} ***")
driver(cmd, ckpts, save_fn)