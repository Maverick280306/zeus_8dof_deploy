"""
The GUI key is made once and then kept.

What a failure would mean: an address that worked a minute ago is refused
after a restart (the operator opens the robot's page and gets a 403), or the
key file is readable by other users on the Pi.
"""

import os
import stat

from zeus_link.gui_key import stable_key


def test_made_once_then_the_same(tmp_path):
    p = str(tmp_path / "zeus" / "gui_key")
    k1 = stable_key(p)
    assert len(k1) >= 12
    assert stable_key(p) == k1          # a second launch: the same address


def test_only_its_owner_can_read_it(tmp_path):
    p = str(tmp_path / "gui_key")
    stable_key(p)
    assert stat.S_IMODE(os.stat(p).st_mode) == 0o600


def test_deleting_the_file_changes_the_key(tmp_path):
    p = str(tmp_path / "gui_key")
    k1 = stable_key(p)
    os.remove(p)
    assert stable_key(p) != k1
