#!/usr/bin/env python3
from pathlib import Path
import subprocess
here=Path(__file__).resolve().parent
root=here.parents[1]
out=here/'generated'
s=(out/'trsm-front48.c').read_text()
marker='    g_align_owner_buffers ='
assert s.count(marker)==1
s=s.replace(marker,'    if(env_int("TRSM_EXPECT_FRONT48",0) && (c.special!=5 || !g_owner_cursor))abort();\n'+marker)
(out/'trsm-front48-audit.c').write_text(s)
driver=subprocess.check_output(['git','show','b138512:probe/v50_20260912/independent-owner-switch.c'],cwd=root).decode()
driver=driver.replace('setenv("TRSM_OWNER",reuse?"0":"1",1);','setenv("TRSM_OWNER","1",1);\n            setenv("TRSM_OWNER_CURSOR","1",1);\n            setenv("TRSM_EXPECT_FRONT48","1",1);')
driver=driver.replace('setenv("TRSM_EXPECT_OWNER",reuse?"0":"1",1);','setenv("TRSM_EXPECT_OWNER","1",1);')
(out/'independent-front48.c').write_text(driver)
