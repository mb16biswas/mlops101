import bentoml
from bentoml.io import NumpyNdarray

reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()
ran_runner = bentoml.sklearn.get("random_reg:latest").to_runner()

reg = bentoml.Service('regression', runners=[ reg_runner , ran_runner ])

input_spec = NumpyNdarray(
    dtype="float",
    shape=(-1, 4) 
)

@reg.api(input=input_spec, output=NumpyNdarray())
def reg_predict(input_arr):
    return reg_runner.predict.run(input_arr)


@reg.api(input=input_spec, output=NumpyNdarray())
def ran_predict(input_arr):
    return ran_runner.predict.run(input_arr)




# import bentoml
# from bentoml.io import NumpyNdarray

# reg_runner = bentoml.sklearn.get("linear_reg:latest").to_runner()

# reg_runner2 = bentoml.sklearn.get("linear_reg2:latest").to_runner()

# svc = bentoml.Service("linear_regression", runners=[reg_runner,reg_runner2])

# input_spec = NumpyNdarray(dtype="int", shape=(-1, 4))


# @svc.api(input=input_spec, output=NumpyNdarray())
# async def predict(input_arr):
#     return await reg_runner.predict.async_run(input_arr)

# @svc.api(input=input_spec, output=NumpyNdarray())
# async def predict2(input_arr):
#     return await reg_runner2.predict.async_run(input_arr)