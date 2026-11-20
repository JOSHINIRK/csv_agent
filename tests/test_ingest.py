import pytest

from csvagent import CSVImportError, load_csv


def roles(ds):
    return {c.name: (c.semantic_type, c.role) for c in ds.schema.columns}


def test_detects_types_and_roles(settings):
    lines = ["order_id,Order Date,Region,Amount,Discount,Returned,Notes"]
    for i in range(80):
        lines.append(
            f"{1000 + i},2024-0{1 + i % 9}-1{i % 9},{['North', 'South', 'East'][i % 3]},"
            f'"${1000 + i * 10:,}.50",{i % 5 * 5}%,{"Yes" if i % 4 == 0 else "No"},'
            f"customer said something long enough to be free text number {i} and more words here"
        )
    ds = load_csv("\n".join(lines).encode(), name="t.csv", settings=settings)
    r = roles(ds)
    assert r["order_id"] == ("identifier", "id")
    assert r["Order Date"] == ("datetime", "time")
    assert r["Region"] == ("categorical", "dimension")
    assert r["Amount"] == ("float", "measure")
    assert ds.df["Amount"].iloc[0] == pytest.approx(1000.5)
    assert r["Discount"][1] == "measure" and ds.df["Discount"].max() == 20
    assert r["Returned"] == ("boolean", "dimension")
    assert r["Notes"] == ("text", "text")


def test_semicolon_cp1252_and_index_column(settings):
    text = ";Ville;Montant\n0;Montréal;1,5\n1;Québec;2\n2;Montréal;3\n"
    ds = load_csv(text.encode("cp1252"), settings=settings)
    assert ds.schema.delimiter == ";"
    assert ds.schema.encoding == "cp1252"
    assert list(ds.df.columns) == ["Ville", "Montant"]
    assert "Montréal" in set(ds.df["Ville"])
    assert any("index column" in w for w in ds.schema.warnings)


def test_missing_markers_and_duplicate_headers(settings):
    ds = load_csv(b"a,a,b\n1,?,x\n2,N/A,y\n3,5,\n", settings=settings)
    assert list(ds.df.columns) == ["a", "a_2", "b"]
    assert ds.df["a_2"].isna().sum() == 2
    assert ds.schema.column("a_2").semantic_type == "integer"


def test_day_first_dates(settings):
    ds = load_csv(b"d,v\n25/03/2024,1\n02/04/2024,2\n13/05/2024,3\n", settings=settings)
    assert ds.df["d"].dt.month.tolist() == [3, 4, 5]


def test_year_column_is_time_and_zip_codes_stay_text(settings):
    rows = "\n".join(f"{2000 + i},0{2100 + i},{1000 * i}" for i in range(20))
    ds = load_csv(f"Year,Zip,Population\n{rows}".encode(), settings=settings)
    assert ds.schema.column("Year").role == "time"
    assert ds.schema.column("Zip").semantic_type != "integer"
    assert ds.schema.column("Population").role == "measure"


def test_empty_file_is_rejected(settings):
    with pytest.raises(CSVImportError):
        load_csv(b"   \n", settings=settings)


def test_samples_schema(sales, census):
    s = roles(sales)
    assert s["Order Date"] == ("datetime", "time")
    assert s["Order ID"][1] == "id"
    assert s["Sales"] == ("float", "measure")
    assert s["Region"] == ("categorical", "dimension")
    c = roles(census)
    assert c["income"] == ("categorical", "dimension")
    assert census.schema.column("workclass").missing > 0  # "?" read as missing
    assert "Dataset 'retail_sales.csv'" in sales.schema.to_prompt()


def test_decimal_commas_and_european_thousands(settings):
    raw = "Date;Rate;Revenue;Units\n2025-01-01;0,505;4585,82;1.200\n2025-01-02;0,431;1.234,50;900\n".encode()
    ds = load_csv(raw, settings=settings)
    assert ds.df["Rate"].tolist() == pytest.approx([0.505, 0.431])
    assert ds.df["Revenue"].tolist() == pytest.approx([4585.82, 1234.5])


def test_us_thousands_still_work(settings):
    ds = load_csv(b'a,b\n"1,234",1.5\n"12,000",2.25\n', settings=settings)
    assert ds.df["a"].tolist() == [1234, 12000]


def test_none_is_a_category_not_missing(settings):
    ds = load_csv(b"Garage,Price\nNone,1\n2-car,2\nNone,3\n,4\n", settings=settings)
    assert ds.df["Garage"].isna().sum() == 1


def test_unique_names_are_dimensions_but_codes_are_ids(settings):
    words = ["Alpha", "Bravo", "Charlie", "Delta", "Echo", "Foxtrot", "Golf", "Hotel", "India"]
    countries = [f"{a} {b}" for a in words for b in words]  # 81 distinct multi-word names
    rows = "\n".join(f"{name},C-{i:03d},{i * 1.5}" for i, name in enumerate(countries))
    ds = load_csv(f"COUNTRY,REF,GDP\n{rows}".encode(), settings=settings)
    assert ds.schema.column("COUNTRY").role == "dimension"
    assert ds.schema.column("REF").role == "id"
