# mukesh_description

The ankle-roll build: the model the deployed policy was trained on. It is
generated, so **edit `config/mukesh_model.yaml`, never the URDF**:

```bash
python3 zeus_description/scripts/clean_urdf.py --config mukesh_description/config/mukesh_model.yaml
```

Ten actuated joints, four spring joints, 11.507 kg, X forward, Y left, Z up,
origin at the hip centre. Two facts in the YAML are assumed, not confirmed, and
both are on the deployment checklist in
[docs/MUKESH_DEPLOYMENT.md](../docs/MUKESH_DEPLOYMENT.md): which way the foot
points (forward = +Y in CAD gives a heel-long foot), and where the IMU sits.
