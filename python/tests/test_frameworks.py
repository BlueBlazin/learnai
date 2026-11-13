import numpy as np
import pytest

from whetstone.judge import CheckFailure, frameworks


def test_float_arrays_become_float32_tensors_and_round_trip(fw):
    x = np.arange(6, dtype=np.float64).reshape(2, 3) / 7
    tensor = fw.tensor(x)
    back = fw.numpy(tensor)
    assert back.shape == (2, 3)
    np.testing.assert_allclose(back, x.astype(np.float32))
    if fw.name == "python":
        assert isinstance(tensor, list) and isinstance(tensor[0][0], float)
    else:
        assert fw.is_tensor(tensor)
        assert back.dtype == np.float32


def test_integer_and_bool_arrays_keep_their_kind(fw):
    ints = fw.numpy(fw.tensor(np.array([1, 2, 3], dtype=np.int64)))
    assert ints.dtype.kind == "i" and ints.tolist() == [1, 2, 3]
    if fw.name == "torch":
        assert ints.dtype == np.int64
    if fw.name in ("jax", "mlx"):
        assert ints.dtype == np.int32
    flags = fw.numpy(fw.tensor(np.array([True, False])))
    assert flags.dtype.kind == "b"


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.int32])
def test_narrow_integer_arrays_keep_their_dtype(fw, dtype):
    if fw.name == "python":
        pytest.skip("lists have no dtype")
    info = np.iinfo(dtype)
    array = np.array([info.min, -1 if info.min < 0 else 1, 0, info.max], dtype=dtype)
    tensor = fw.tensor(array)
    assert str(tensor.dtype).split(".")[-1] == np.dtype(dtype).name  # the tensor itself
    back = fw.numpy(tensor)
    assert back.dtype == dtype and back.tolist() == array.tolist()
    # The recursive conversion that ctx.call uses follows the same rule.
    (nested,) = fw.to_framework([array])
    assert fw.numpy(nested).dtype == dtype


def test_64_bit_integers_become_the_framework_default(fw):
    if fw.name == "python":
        pytest.skip("lists have no dtype")
    wide = fw.name == "torch"
    assert fw.numpy(fw.tensor(np.arange(3))).dtype == (np.int64 if wide else np.int32)
    unsigned = fw.numpy(fw.tensor(np.arange(3, dtype=np.uint64)))
    assert unsigned.dtype == (np.uint64 if wide else np.uint32)
    assert fw.numpy(fw.tensor(np.arange(3, dtype=np.int8), dtype=np.int64)).dtype.kind == "i"


def test_explicit_dtype_is_respected(fw):
    if fw.name == "python":
        pytest.skip("lists have no dtype")
    assert fw.numpy(fw.tensor(np.array([1.0, 2.0]), dtype=np.float16)).dtype == np.float16


def test_recursive_conversion_leaves_other_values_alone(fw):
    tree = {
        "x": np.ones((2, 2)),
        "items": [np.zeros(3), 5, "text"],
        "pair": (np.float32(2.5), None),
    }
    converted = fw.to_framework(tree)
    assert converted["items"][1:] == [5, "text"]
    assert converted["pair"] == (2.5, None) and isinstance(converted["pair"][0], float)
    back = fw.to_numpy(converted)
    np.testing.assert_array_equal(np.asarray(back["x"]), np.ones((2, 2)))
    assert back["items"][1:] == [5, "text"]
    if fw.name != "python":
        assert fw.is_tensor(converted["x"]) and fw.is_tensor(converted["items"][0])
        assert isinstance(back["x"], np.ndarray)


def test_inputs_are_copied_not_aliased(fw):
    if fw.name != "torch":
        pytest.skip("only torch can alias numpy memory")
    x = np.zeros(3, dtype=np.float32)
    fw.tensor(x).add_(1.0)
    assert x.tolist() == [0.0, 0.0, 0.0]


def test_wrong_tensor_type_is_reported_clearly(fw):
    if fw.name == "python":
        assert fw.to_numpy(np.ones(2)).tolist() == [1.0, 1.0]
        return
    with pytest.raises(CheckFailure) as failure:
        fw.to_numpy(np.ones(2), what="`softmax`")
    assert failure.value.message == (
        f"`softmax` returned a numpy.ndarray, expected a {fw.tensor_type_name}"
    )
    other = frameworks.get("mlx" if fw.name == "torch" else "torch")
    with pytest.raises(CheckFailure, match="expected a"):
        fw.to_numpy((other.tensor(np.ones(2)),))


def test_sync_accepts_nested_results(fw):
    value = fw.tensor(np.ones((4, 4)))
    fw.sync(value, [value, {"k": (value, 3)}], None)


def test_kit_objects_are_rebound(fw):
    class Kit:
        def __init__(self, bound):
            self.fw = bound

        def with_framework(self, bound):
            return Kit(bound)

    here = Kit(fw)
    assert fw.to_framework([here])[0] is here  # already bound: untouched
    fresh = fw.to_framework([here], fresh_kit=True)[0]
    assert fresh is not here and fresh.fw is fw  # reference calls get their own twin
    elsewhere = Kit(object())
    assert fw.to_framework({"m": elsewhere})["m"].fw is fw


def test_unknown_framework():
    with pytest.raises(ValueError):
        frameworks.get("tensorflow")
