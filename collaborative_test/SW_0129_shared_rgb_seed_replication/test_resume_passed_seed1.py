import unittest
from collaborative_test.SW_0129_shared_rgb_seed_replication import coordinator
from collaborative_test.SW_0129_shared_rgb_seed_replication import resume_passed_seed1 as resume


def parent_fixture():
    tasks = {}
    for task in coordinator.task_plan():
        if task['task_id'] == 'sw0129_preflight_s1':
            row = {'status': 'passed', 'artifact_valid': True, 'child_pid': 201}
        elif task['task_id'] == 'sw0129_preflight_s2':
            row = {'status': 'failed', 'artifact_valid': False, 'returncode': 1,
                   'child_pid': 202, 'log_path': (resume.ARCHIVE / 'sw0129_preflight_s2.log').as_posix()}
        elif task['stage'] in ('train', 'evaluate'):
            row = {'status': 'blocked_preflight_failure'}
        else:
            row = {'status': 'queued'}
        tasks[task['task_id']] = row
    return {'experiment': 'SW0129', 'status': 'preflight_failed', 'supervisor_pid': 200,
            'tasks': tasks}


class ResumeSeed1Tests(unittest.TestCase):
    def test_accepts_only_passed_seed1_and_terminal_scientific_seed2_failure(self):
        state = parent_fixture()
        self.assertTrue(resume.validate_parent_state(
            state, lambda _pid: True,
            lambda _path: 'AssertionError: mean hard-partition row scrambling did not increase reconstruction loss'))
        tasks = resume.resume_tasks()
        self.assertEqual({(t['stage'], t['seed'], t['arm']) for t in tasks},
                         {('train', 1, 'control'), ('train', 1, 'candidate'),
                          ('evaluate', 1, 'control'), ('evaluate', 1, 'candidate')})

    def test_rejects_nonterminal_or_different_failure_and_live_pid(self):
        state = parent_fixture()
        with self.assertRaisesRegex(ValueError, 'registered row-scramble'):
            resume.validate_parent_state(state, lambda _pid: True, lambda _path: 'CUDA runtime error')
        state = parent_fixture(); state['tasks']['sw0129_preflight_s1']['status'] = 'failed'
        with self.assertRaisesRegex(ValueError, 'seed-1 preflight'):
            resume.validate_parent_state(state, lambda _pid: True, lambda _path: 'row scrambling')
        state = parent_fixture()
        with self.assertRaisesRegex(RuntimeError, 'live/unknown'):
            resume.validate_parent_state(state, lambda pid: pid != 202,
                                         lambda _path: 'row scrambling')

    def test_refuses_other_parent_terminal_state(self):
        state = parent_fixture(); state['status'] = 'complete'
        with self.assertRaisesRegex(ValueError, 'preflight_failed'):
            resume.validate_parent_state(state, lambda _pid: True, lambda _path: '')


if __name__ == '__main__': unittest.main()
