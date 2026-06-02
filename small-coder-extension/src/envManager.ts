import * as path from 'path';
import { spawn, spawnSync } from 'child_process';
import { OutputChannel } from 'vscode';
import { platform } from 'os';

export async function findPython(): Promise<string | undefined> {
  const candidates = platform() === 'win32' ? ['py', 'python', 'python3'] : ['python3', 'python'];

  for (const candidate of candidates) {
    const result = spawnSync(candidate, ['--version'], { encoding: 'utf8' });
    if (result.status === 0 && result.stdout.trim().length > 0) {
      return candidate;
    }
  }

  return undefined;
}

export async function createVirtualEnv(envPath: string, pythonPath: string, output: OutputChannel): Promise<void> {
  output.appendLine(`Creating virtual environment at ${envPath} with ${pythonPath}`);
  await runProcess(pythonPath, ['-m', 'venv', envPath], output);
}

export async function installRequirements(
  envPath: string,
  requirementsPath: string,
  output: OutputChannel,
): Promise<void> {
  const pipPath = getVenvExecutable(envPath, 'pip');
  output.appendLine(`Installing requirements from ${requirementsPath}`);
  await runProcess(pipPath, ['install', '-r', requirementsPath], output);
}

export function getVenvPython(envPath: string): string {
  return getVenvExecutable(envPath, 'python');
}

function getVenvExecutable(envPath: string, executableName: string): string {
  if (platform() === 'win32') {
    return path.join(envPath, 'Scripts', `${executableName}.exe`);
  }
  return path.join(envPath, 'bin', executableName);
}

function runProcess(command: string, args: string[], output: OutputChannel): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { stdio: 'pipe' });

    child.stdout.on('data', (data) => output.appendLine(data.toString()));
    child.stderr.on('data', (data) => output.appendLine(data.toString()));

    child.on('error', (err) => reject(err));
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`${command} ${args.join(' ')} failed with exit code ${code}`));
      } else {
        resolve();
      }
    });
  });
}
