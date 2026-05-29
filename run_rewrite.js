const { spawnSync } = require('child_process');
const env = { ...process.env, FILTER_BRANCH_SQUELCH_WARNING: '1' };
const cmd = 'git';
const args = [
    'filter-branch', '-f', '--env-filter',
    'export GIT_AUTHOR_NAME="Đoàn Anh Hùng"; export GIT_AUTHOR_EMAIL="hungdoan772@gmail.com"; export GIT_COMMITTER_NAME="Đoàn Anh Hùng"; export GIT_COMMITTER_EMAIL="hungdoan772@gmail.com";',
    '--', '--all'
];
console.log("Running:", cmd, args.join(" "));
const result = spawnSync(cmd, args, { env, encoding: 'utf-8' });
console.log("STDOUT:", result.stdout);
console.log("STDERR:", result.stderr);
if (result.status !== 0) {
    console.error("FAILED with status", result.status);
    process.exit(1);
}
