from pydantic import BaseModel, Field, validator, confloat, conint



class SpinQubit(BaseModel):
    n_q: int = conint(ge=1, le=20)
    Tq: float = confloat(gt=0.0, le=300)
    f: float = Field(39.33e9, alias='f_in_GHz')
    rabi: float = Field(0.5e6, alias='rabi_in_MHz')
    atts_list: list = []
    stages_ts: list = []
    silicon_abs: float = confloat(ge=0.0, le=1.0)

    @property
    def time(self) -> float:
        return 1e-6 / (2 * self.rabi)

    @property
    def gamma(self) -> float:
        return 0.58 * 1e-8
    

if __name__ == "__main__":
    print("runs")
