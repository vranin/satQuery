"""Search Copernicus Data Space for a later Sentinel-2 product."""
from __future__ import annotations

import getpass
import glob
import os
from datetime import datetime, timezone

import requests
import rasterio
from rasterio.warp import transform_bounds

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CATALOG_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
PATCH_DIR = "data/bigearthnet/S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57"


def main() -> None:
    username = os.getenv("CDSE_USERNAME") or input("Copernicus username/email: ")
    password = os.getenv("CDSE_PASSWORD") or getpass.getpass("Copernicus password: ")
    patch_path = glob.glob(f"{PATCH_DIR}/*_B04.tif")[0]

    with rasterio.open(patch_path) as src:
        left, bottom, right, top = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    polygon = f"POLYGON(({left} {bottom},{right} {bottom},{right} {top},{left} {top},{left} {bottom}))"
    token_response = requests.post(
        TOKEN_URL,
        data={
            "client_id": "cdse-public",
            "grant_type": "password",
            "username": username,
            "password": password,
        },
        timeout=30,
    )
    token_response.raise_for_status()

    start = datetime(2017, 6, 13, tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
    query_filter = (
        "Collection/Name eq 'SENTINEL-2' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{polygon}') and "
        f"ContentDate/Start gt {start}"
    )
    response = requests.get(
        CATALOG_URL,
        params={"$filter": query_filter, "$orderby": "ContentDate/Start asc", "$top": "10"},
        headers={"Authorization": f"Bearer {token_response.json()['access_token']}"},
        timeout=60,
    )
    response.raise_for_status()
    products = response.json().get("value", [])
    if not products:
        print("No later Sentinel-2 products found for this footprint.")
        return
    for product in products:
        print(product.get("Name"), product.get("ContentDate"), product.get("Online"), product.get("Id"))


if __name__ == "__main__":
    main()