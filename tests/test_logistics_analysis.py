import pandas as pd
import pytest

from logistics_analysis import (
    calculate_kpis,
    clean_data,
    delivery_performance_by_mode,
    evaluate_classifier,
    identify_columns,
    inspect_data,
    late_delivery_target,
    run_analysis,
)


@pytest.fixture
def sample_frame():
    return pd.DataFrame(
        {
            "Delivery Status": ["Late delivery", "Shipping on time", "Advance shipping"],
            "Shipping Mode": ["Standard Class", "First Class", "Same Day"],
            "Shipping Cost": ["10.50", "8.00", "20.00"],
            "Days for Shipping (Real)": ["5", "3", "1"],
            "Days for Shipment (Scheduled)": ["4", "3", "1"],
            "Order Date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )


def test_kpis_calculate_rates_cost_time_and_modes(sample_frame):
    kpis = calculate_kpis(sample_frame)

    assert kpis["total_shipments"] == 3
    assert kpis["late_delivery_rate"] == pytest.approx(1 / 3)
    assert kpis["on_time_delivery_rate"] == pytest.approx(2 / 3)
    assert kpis["average_shipping_cost"] == pytest.approx(38.5 / 3)
    assert kpis["average_delivery_time_days"] == pytest.approx(3)
    assert kpis["shipment_count_by_shipping_mode"] == {
        "First Class": 1,
        "Same Day": 1,
        "Standard Class": 1,
    }


def test_missing_columns_are_reported_as_unavailable():
    frame = pd.DataFrame({"unrelated": [1, 2, 3]})

    kpis = calculate_kpis(frame)

    assert kpis["total_shipments"] == 3
    assert kpis["late_delivery_rate"] is None
    assert kpis["on_time_delivery_rate"] is None
    assert kpis["average_shipping_cost"] is None
    assert kpis["average_delivery_time_days"] is None
    assert kpis["shipment_count_by_shipping_mode"] == {}
    assert all(value is None for value in identify_columns(frame).values())


def test_cleaning_parses_values_and_retains_invalid_and_duplicate_rows():
    frame = pd.DataFrame(
        {
            "Shipping Cost": ["12.5", "not a cost", "12.5"],
            "Order Date": ["2025-01-01", "not a date", "2025-01-01"],
        }
    )

    cleaned, notes = clean_data(frame)

    assert len(cleaned) == 3
    assert cleaned["Shipping Cost"].iloc[0] == pytest.approx(12.5)
    assert pd.isna(cleaned["Shipping Cost"].iloc[1])
    assert pd.isna(cleaned["Order Date"].iloc[1])
    assert any("duplicate row(s) were identified but not removed" in note for note in notes)
    assert any("1 non-empty value(s) could not be parsed" in note for note in notes)


def test_late_target_prefers_observed_status_and_excludes_canceled_records():
    frame = pd.DataFrame(
        {
            "Delivery Delay Days": [2, 0, 5],
            "Delivery Status": ["Shipping on time", "Late delivery", "Shipping canceled"],
            "Late_delivery_risk": [1, 0, 0],
        }
    )

    target = late_delivery_target(frame)

    assert target.iloc[0] == 0
    assert target.iloc[1] == 1
    assert pd.isna(target.iloc[2])


def test_dataco_column_aliases_and_mode_summary():
    frame = pd.DataFrame(
        {
            "order date (DateOrders)": ["1/31/2018 22:56", "2/1/2018 22:56"],
            "Delivery Status": ["Late delivery", "Advance shipping"],
            "Shipping Mode": ["Standard Class", "First Class"],
            "Days for shipping (real)": [5, 2],
            "Days for shipment (scheduled)": [4, 3],
        }
    )

    columns = identify_columns(frame)
    kpis = calculate_kpis(frame)
    by_mode = delivery_performance_by_mode(frame).set_index("shipping_mode")

    assert columns["shipment_date"] == "order date (DateOrders)"
    assert columns["delivery_time_days"] == "Days for shipping (real)"
    assert kpis["late_delivery_rate"] == pytest.approx(0.5)
    assert kpis["average_shipping_cost"] is None
    assert by_mode.loc["Standard Class", "late_delivery_rate"] == 1
    assert by_mode.loc["First Class", "late_delivery_rate"] == 0


def test_inspection_sample_suppresses_customer_fields():
    frame = pd.DataFrame(
        {
            "Delivery Status": ["Late delivery"],
            "Shipping Mode": ["Standard Class"],
            "Customer Fname": ["Private Name"],
            "Customer Street": ["Private Street"],
            "Customer Email": ["private@example.test"],
        }
    )

    summary = inspect_data(frame, identify_columns(frame))

    assert "Late delivery" in summary
    assert "Private Name" not in summary
    assert "Private Street" not in summary
    assert "private@example.test" not in summary


def test_classifier_trains_only_from_non_leaking_predictors():
    frame = pd.read_csv("data/synthetic_logistics_sample.csv")

    report, metrics = evaluate_classifier(frame)

    assert metrics is not None
    assert "Predictors:" in report
    assert "Delivery Delay Days" not in report
    assert "Days for Shipping (Real)" not in report
    assert 0 <= metrics["accuracy"] <= 1


def test_analysis_writes_expected_reports_and_skips_unsupported_charts(tmp_path):
    input_path = tmp_path / "small.csv"
    pd.DataFrame({"unrelated": [1, 2, 3]}).to_csv(input_path, index=False)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    (output_dir / "shipping_cost_analysis.png").write_bytes(b"stale synthetic chart")
    project_report = tmp_path / "reports" / "analysis.md"

    generated = run_analysis(
        input_path,
        output_dir,
        data_kind="synthetic",
        project_report_path=project_report,
    )

    names = {path.name for path in generated}
    assert "data_summary.txt" in names
    assert "kpi_summary.csv" in names
    assert "analysis_report.md" in names
    assert "delivery_status.png" not in names
    assert not (output_dir / "shipping_cost_analysis.png").exists()
    assert (output_dir / "clustering_summary.txt").is_file()
    assert "synthetic testing dataset" in project_report.read_text(encoding="utf-8")
