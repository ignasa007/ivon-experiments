fns = [
    "gradient.py",
    "momentum.py",
    "expavgsq_baseline.py",
    "expavgsq_25k.py",
    # "expavgsq_xent.py",     # This needs editting the threshold for filtering points
    "expavgsq_fmnist.py",
    "expavgsq_cifar10.py",
    "expavgsq_cifar100.py",
    "expavgsq_wide.py",
    "expavgsq_deep.py",
    "expavgsq_tanh.py",
    "expavgsq_beta2.py",
    "expavgsq_eps.py",
]

for fn in fns:
    print(f"*** Running {fn} ***")
    exec(open(fn).read())