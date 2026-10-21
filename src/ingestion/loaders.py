"""
Load raw energy CSVs into LangChain Documents.

Each document is one day of data for one entity (grid node or household).
Metadata is attached so the agent can filter by entity/date later.
"""
import pandas as pd
from langchain_core.documents import Document


def load_ev_grid_documents(csv_path: str) -> list[Document]:
    """
    Convert EV grid features CSV into daily Documents.

    Each Document covers one grid node on one date, with hourly readings
    formatted as readable text.
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour_str'] = df['timestamp'].dt.strftime('%H:%M')

    docs = []
    for (node, date), group in df.groupby(['grid_node', 'date']):
        lines = [
            f"Grid Node: {node}",
            f"Date: {date}",
            f"Total Energy: {group['total_energy_kwh'].sum():.2f} kWh",
            f"Charging Hours: {len(group)}",
            "",
            "Hourly readings:",
        ]
        for _, row in group.iterrows():
            lines.append(
                f"  {row['hour_str']} — "
                f"Energy: {row['total_energy_kwh']:.2f} kWh, "
                f"Active chargers: {int(row['active_chargers'])}, "
                f"Unique vehicles: {int(row['unique_vehicles'])}"
            )

        content = "\n".join(lines)
        metadata = {
            "source": "ev_grid",
            "grid_node": node,
            "date": str(date),
            "total_energy_kwh": float(group['total_energy_kwh'].sum()),
            "n_hours": int(len(group)),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs


def load_solar_documents(csv_path: str) -> list[Document]:
    """
    Convert solar features CSV into daily Documents for Palm Springs.

    Each Document is one calendar day with hourly GHI, cloud, and CSI values.
    """
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour_str'] = df['timestamp'].dt.strftime('%H:%M')

    docs = []
    for date, group in df.groupby('date'):
        peak_ghi = group['ghi'].max()
        peak_hour = group.loc[group['ghi'].idxmax(), 'hour_str']
        mean_cloud = group['total_cloud_cover'].mean()

        lines = [
            "Location: Palm Springs, CA",
            f"Date: {date}",
            f"Peak GHI: {peak_ghi:.1f} W/m² at {peak_hour}",
            f"Mean Cloud Cover: {mean_cloud:.1f}%",
            "",
            "Hourly readings (time, GHI, cloud, CSI):",
        ]
        for _, row in group.iterrows():
            lines.append(
                f"  {row['hour_str']} — "
                f"GHI: {row['ghi']:.1f} W/m², "
                f"Cloud: {row['total_cloud_cover']:.0f}%, "
                f"CSI: {row['csi']:.2f}"
            )

        content = "\n".join(lines)
        metadata = {
            "source": "solar",
            "location": "Palm Springs",
            "date": str(date),
            "peak_ghi": float(peak_ghi),
            "mean_cloud_cover": float(mean_cloud),
        }
        docs.append(Document(page_content=content, metadata=metadata))

    return docs