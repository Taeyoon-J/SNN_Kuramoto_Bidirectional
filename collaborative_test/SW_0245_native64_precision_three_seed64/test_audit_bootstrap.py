"""Regression: absolute script execution does not put the repository on sys.path."""
import subprocess,sys
from pathlib import Path
from . import completion_audit_recovery_v2 as recovery


def test_absolute_auditor_can_import_repository_from_foreign_cwd(tmp_path):
    script=Path(recovery.__file__).resolve();root=script.parents[2]
    code="import importlib.util,sys; spec=importlib.util.spec_from_file_location('standalone_auditor',sys.argv[1]);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.activate_repo(sys.argv[2]);import collaborative_test;print('repository_import_ok')"
    result=subprocess.run([sys.executable,'-I','-B','-c',code,str(script),str(root)],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
    assert result.stdout.strip()=='repository_import_ok'
