from app.config import Settings


def test_required_settings_are_required_and_policy_settings_have_defaults() -> None:
    optional_fields = {"grade_low", "grade_high", "cache_ttl_seconds"}
    assert all(
        field.is_required()
        for name, field in Settings.model_fields.items()
        if name not in optional_fields
    )
    assert Settings.model_fields["grade_low"].default == 0.35
    assert Settings.model_fields["grade_high"].default == 0.75
    assert Settings.model_fields["cache_ttl_seconds"].default == 3600
