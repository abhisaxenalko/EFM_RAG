from app.config import Settings


def test_all_settings_are_required() -> None:
    assert all(field.is_required() for field in Settings.model_fields.values())
