import json
from typing import Dict


def make_class_table(metrics_json: Dict) -> str:
    """
    Generate HTML table from metrics JSON containing class AP values.

    Args:
        metrics_json: Dict containing val_metrics with segm/AP_{class_id} entries

    Returns:
        HTML string containing formatted table
    """

    table = """<div class="class-table" markdown>
  <style>
    .class-table {
      max-height: 500px;
      overflow-y: auto;
      padding: 1rem;
      margin: 1rem 0;
      background: rgba(0,0,0,0.05);
      width: 95%;
      margin-left: auto;
      margin-right: auto;
    }
    .class-table table {
      width: 100%;
    }
    .class-table thead {
      position: sticky;
      top: 0;
      background: #2b2b2b;
      z-index: 1;
    }
  </style>
<table>
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Segmentation AP</th>
    </tr>
  </thead>
  <tbody>"""

    val_metrics = metrics_json["val_metrics"]
    print(val_metrics)
    classes = metrics_json["classes"]

    for class_id, class_name in enumerate(classes, start=1):
        ap_key = f"sem_seg/IoU-{class_name}"
        ap_value = val_metrics.get(ap_key, 0.0)

        table += f"""
    <tr>
      <td>{class_id}</td>
      <td>{class_name}</td>
      <td>{ap_value:.6f}</td>
    </tr>"""

    table += """
  </tbody>
</table>

</div>"""

    return table


if __name__ == "__main__":
    with open("metrics+.json", "r") as f:
        metrics_json = json.load(f)
    print(make_class_table(metrics_json))
