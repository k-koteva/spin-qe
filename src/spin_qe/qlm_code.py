from qat.lang import CNOT, H, qrout


@qrout
def bell_pair():
    H(0)
    CNOT(0, 1)


result = bell_pair().run()

for sample in result:
    print(f"State {sample.state} amplitude {sample.amplitude}")
