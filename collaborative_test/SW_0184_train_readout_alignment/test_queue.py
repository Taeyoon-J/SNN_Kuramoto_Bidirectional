import unittest
from unittest import mock
from collaborative_test.SW_0184_train_readout_alignment import queue

class AlignmentQueue(unittest.TestCase):
    def test_actual_worker_module_and_seed_assignment(self):
        command=queue.worker_command(2,3)
        self.assertEqual(command[command.index('-m')+1],queue.MODULE)
        self.assertEqual(command[-4:],['--seed','2','--physical-gpu','3'])
        with self.assertRaises(ValueError):queue.worker_command(3,0)
        with self.assertRaises(ValueError):queue.worker_command(0,4)

    def test_cli_help(self):
        with self.assertRaises(SystemExit) as result:queue._worker_entry(['--help'])
        self.assertEqual(result.exception.code,0)

    def test_terminal_status_published_once_without_duplicate_keyword(self):
        with mock.patch.object(queue,'_write_state') as write:
            queue._publish_complete_state(1,{'seed':1,'status':'complete','optimizer_updates':0})
        write.assert_called_once_with(1,'complete',optimizer_updates=0)

if __name__=='__main__':unittest.main()
