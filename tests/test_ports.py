from dataclasses import dataclass
from typing import Optional

import pytest
from pytest_mock import MockerFixture

from focoos.ports import (
    DictClass,
    GPUDevice,
    GPUInfo,
    ModelExtension,
    RuntimeType,
    SystemInfo,
)


@dataclass
class TestDictDataclass(DictClass):
    """Dataclass di test per verificare il comportamento di DictClass"""

    name: str
    value: int
    optional_field: Optional[str] = None
    default_field: str = "default"


class TestDictClass:
    """Test suite per la classe DictClass"""

    def test_dataclass_initialization_with_required_fields(self):
        """Verifica che una dataclass che eredita da DictClass si inizializzi correttamente"""
        test_obj = TestDictDataclass(name="test", value=42)

        assert test_obj.name == "test"
        assert test_obj.value == 42
        assert test_obj.optional_field is None
        assert test_obj.default_field == "default"

    def test_dataclass_initialization_with_all_fields(self):
        """Verifica che una dataclass si inizializzi con tutti i campi specificati"""
        test_obj = TestDictDataclass(name="complete_test", value=100, optional_field="optional", default_field="custom")

        assert test_obj.name == "complete_test"
        assert test_obj.value == 100
        assert test_obj.optional_field == "optional"
        assert test_obj.default_field == "custom"

    def test_dict_like_access_with_string_keys(self):
        """Verifica che l'oggetto possa essere usato come un dizionario con chiavi stringa"""
        test_obj = TestDictDataclass(name="dict_test", value=123)

        assert test_obj["name"] == "dict_test"
        assert test_obj["value"] == 123
        assert test_obj["optional_field"] is None
        assert test_obj["default_field"] == "default"

    def test_dict_like_access_with_integer_indices(self):
        """Verifica che l'oggetto supporti l'accesso tramite indici numerici"""
        test_obj = TestDictDataclass(name="index_test", value=456)
        tuple_representation = test_obj.to_tuple()

        # Verifica che l'accesso tramite indice restituisca elementi della tupla
        assert len(tuple_representation) > 0
        assert test_obj[0] == tuple_representation[0]
        if len(tuple_representation) > 1:
            assert test_obj[1] == tuple_representation[1]

    def test_to_tuple_method(self):
        """Verifica che il metodo to_tuple restituisca una tupla con tutti i valori non None"""
        test_obj = TestDictDataclass(name="tuple_test", value=789, optional_field="present")
        result_tuple = test_obj.to_tuple()

        assert isinstance(result_tuple, tuple)
        assert len(result_tuple) == 4  # name, value, optional_field, default_field
        assert "tuple_test" in result_tuple
        assert 789 in result_tuple
        assert "present" in result_tuple
        assert "default" in result_tuple

    def test_to_tuple_with_none_values(self):
        """Verifica che to_tuple non includa i valori None"""
        test_obj = TestDictDataclass(name="none_test", value=0, optional_field=None)
        result_tuple = test_obj.to_tuple()

        assert isinstance(result_tuple, tuple)
        assert "none_test" in result_tuple
        assert 0 in result_tuple
        assert None not in result_tuple
        assert "default" in result_tuple

    def test_setattr_updates_dict_and_attribute(self):
        """Verifica che __setattr__ aggiorni sia l'attributo che l'entry del dizionario"""
        test_obj = TestDictDataclass(name="setattr_test", value=111)

        # Modifica un valore esistente
        test_obj.name = "updated_name"
        assert test_obj.name == "updated_name"
        assert test_obj["name"] == "updated_name"

        # Modifica un valore opzionale
        test_obj.optional_field = "new_optional"
        assert test_obj.optional_field == "new_optional"
        assert test_obj["optional_field"] == "new_optional"

    def test_setitem_updates_dict_and_attribute(self):
        """Verifica che __setitem__ aggiorni sia l'entry del dizionario che l'attributo"""
        test_obj = TestDictDataclass(name="setitem_test", value=222)

        # Modifica tramite accesso a dizionario
        test_obj["name"] = "dict_updated_name"
        assert test_obj.name == "dict_updated_name"
        assert test_obj["name"] == "dict_updated_name"

        test_obj["value"] = 999
        assert test_obj.value == 999
        assert test_obj["value"] == 999

    def test_dictionary_behavior_inheritance(self):
        """Verifica che l'oggetto si comporti come un OrderedDict"""
        test_obj = TestDictDataclass(name="dict_behavior", value=333)

        # Test keys(), values(), items()
        keys = list(test_obj.keys())
        values = list(test_obj.values())
        items = list(test_obj.items())

        assert "name" in keys
        assert "value" in keys
        assert "dict_behavior" in values
        assert 333 in values
        assert ("name", "dict_behavior") in items
        assert ("value", 333) in items

    def test_post_init_populates_dict_from_dataclass_fields(self):
        """Verifica che __post_init__ popoli correttamente il dizionario dai campi della dataclass"""
        test_obj = TestDictDataclass(name="post_init_test", value=444)

        # Verifica che tutti i campi siano presenti nel dizionario
        expected_fields = ["name", "value", "optional_field", "default_field"]
        for field in expected_fields:
            assert field in test_obj

        # Verifica che i valori corrispondano
        assert test_obj["name"] == "post_init_test"
        assert test_obj["value"] == 444
        assert test_obj["optional_field"] is None
        assert test_obj["default_field"] == "default"

    def test_reduce_method_for_serialization(self):
        """Verifica che __reduce__ permetta la serializzazione corretta dell'oggetto"""
        test_obj = TestDictDataclass(name="reduce_test", value=555)

        # Test del metodo __reduce__
        reduce_result = test_obj.__reduce__()

        assert isinstance(reduce_result, tuple)
        assert len(reduce_result) == 3

        constructor, args, state = reduce_result
        assert constructor == TestDictDataclass.__new__
        assert args == (TestDictDataclass,)
        assert isinstance(state, dict)
        assert "name" in state
        assert "value" in state

    def test_getitem_with_invalid_key_raises_error(self):
        """Verifica che l'accesso con chiave inesistente sollevi un'eccezione"""
        test_obj = TestDictDataclass(name="error_test", value=666)

        with pytest.raises(KeyError):
            _ = test_obj["nonexistent_key"]

    def test_getitem_with_invalid_index_raises_error(self):
        """Verifica che l'accesso con indice non valido sollevi un'eccezione"""
        test_obj = TestDictDataclass(name="index_error_test", value=777)

        with pytest.raises(IndexError):
            _ = test_obj[10]  # Indice fuori range

    @pytest.mark.parametrize(
        "name,value,optional_field,expected_length",
        [
            ("test1", 1, None, 4),
            ("test2", 2, "optional", 4),
            ("test3", 3, "another", 4),
        ],
    )
    def test_parametrized_dict_creation(
        self, name: str, value: int, optional_field: Optional[str], expected_length: int
    ):
        """Test parametrizzato per verificare la creazione di oggetti DictClass con diversi parametri"""
        test_obj = TestDictDataclass(name=name, value=value, optional_field=optional_field)

        assert len(test_obj) == expected_length
        assert test_obj.name == name
        assert test_obj.value == value
        assert test_obj.optional_field == optional_field

    def test_none_value_handling_in_setattr(self):
        """Verifica che __setattr__ gestisca correttamente i valori None"""
        test_obj = TestDictDataclass(name="none_handling", value=888)

        # Imposta un valore a None
        test_obj.optional_field = None
        assert test_obj.optional_field is None
        assert test_obj["optional_field"] is None

        # Riporta il valore a un valore non None
        test_obj.optional_field = "not_none_anymore"
        assert test_obj.optional_field == "not_none_anymore"
        assert test_obj["optional_field"] == "not_none_anymore"


def test_pretty_print_with_system_info(mocker: MockerFixture):
    """Verifica che pretty_print formatti correttamente tutte le informazioni di sistema"""

    gpu_devices = [
        GPUDevice(
            gpu_id=0,
            gpu_name="NVIDIA GTX 1080",
            gpu_memory_total_gb=8.0,
            gpu_memory_used_percentage=70.0,
            gpu_temperature=65.0,
            gpu_load_percentage=80.0,
        )
    ]
    gpu_info = GPUInfo(gpu_count=1, gpu_driver="NVIDIA", gpu_cuda_version="11.2", devices=gpu_devices)

    system_info = SystemInfo(
        focoos_host="localhost",
        system="Linux",
        system_name="TestSystem",
        cpu_type="Intel",
        cpu_cores=8,
        memory_gb=16.0,
        memory_used_percentage=50.0,
        available_onnx_providers=["provider1", "provider2"],
        disk_space_total_gb=500.0,
        disk_space_used_percentage=60.0,
        packages_versions={"pytest": "6.2.4", "pydantic": "1.8.2"},
        gpu_info=gpu_info,
        environment={"FOCOOS_LOG_LEVEL": "DEBUG", "LD_LIBRARY_PATH": "/usr/local/cuda/lib64"},
    )

    system_info.pprint()


@pytest.mark.parametrize(
    "runtime_type,expected_format",
    [
        (RuntimeType.ONNX_CUDA32, ModelExtension.ONNX),
        (RuntimeType.ONNX_TRT32, ModelExtension.ONNX),
        (RuntimeType.ONNX_TRT16, ModelExtension.ONNX),
        (RuntimeType.ONNX_CPU, ModelExtension.ONNX),
        (RuntimeType.ONNX_COREML, ModelExtension.ONNX),
        (RuntimeType.TORCHSCRIPT_32, ModelExtension.TORCHSCRIPT),
    ],
)
def test_model_format_from_runtime_type(runtime_type, expected_format):
    """Test that from_runtime_type returns correct ModelFormat for each RuntimeType"""
    assert ModelExtension.from_runtime_type(runtime_type) == expected_format


def test_model_format_from_runtime_type_invalid():
    """Test that from_runtime_type raises ValueError for invalid runtime type"""
    with pytest.raises(ValueError, match="Invalid runtime type:.*"):
        ModelExtension.from_runtime_type("invalid_runtime")  # type: ignore
