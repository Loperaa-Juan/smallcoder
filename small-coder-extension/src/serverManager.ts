import * as net from 'net';
import * as path from 'path';
import * as http from 'http';
import { ChildProcess, spawn } from 'child_process';
import { OutputChannel } from 'vscode';
import { getVenvPython } from './envManager';
import { ModelInfo } from './modelManager';

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.listen(0, '127.0.0.1');
    server.on('listening', () => {
      const address = server.address();
      if (address && typeof address === 'object') {
        const port = address.port;
        server.close(() => resolve(port));
      } else {
        server.close(() => reject(new Error('Unable to determine free port')));
      }
    });
    server.on('error', reject);
  });
}

function requestJson<T>(url: string, data?: unknown): Promise<T> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const body = data ? JSON.stringify(data) : undefined;
    const options: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port,
      path: parsed.pathname,
      method: data ? 'POST' : 'GET',
      headers: {
        'Content-Type': 'application/json',
        ...(body ? { 'Content-Length': Buffer.byteLength(body) } : {}),
      },
    };

    const req = http.request(options, (res) => {
      let raw = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => (raw += chunk));
      res.on('end', () => {
        try {
          const parsedJson = JSON.parse(raw);
          resolve(parsedJson as T);
        } catch (error) {
          reject(new Error(`Invalid JSON response: ${error}`));
        }
      });
    });

    req.on('error', reject);
    if (body) {
      req.write(body);
    }
    req.end();
  });
}

function waitForServer(port: number, timeout = 20000): Promise<void> {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const interval = setInterval(async () => {
      try {
        await requestJson<{ status: string }>(`http://127.0.0.1:${port}/health`);
        clearInterval(interval);
        resolve();
      } catch {
        if (Date.now() - start > timeout) {
          clearInterval(interval);
          reject(new Error('SmallCoder server did not become ready in time'));
        }
      }
    }, 500);
  });
}

export class PythonServerManager {
  private process: ChildProcess | null = null;
  private port: number | null = null;
  private currentModel: string | null = null;
  private currentDevice: string | null = null;
  private output: OutputChannel;
  private extensionPath: string;

  constructor(output: OutputChannel, extensionPath: string) {
    this.output = output;
    this.extensionPath = extensionPath;
  }

  public async startServer(envPath: string, model: ModelInfo | string, device: string): Promise<number | undefined> {
    const modelArg = typeof model === 'string' ? model : model.id;

    // Reuse existing server if same model/device and still healthy
    if (this.process && !this.process.killed && this.currentModel === modelArg && this.currentDevice === device && this.port) {
      try {
        await requestJson<{ status: string }>(`http://127.0.0.1:${this.port}/health`);
        this.output.appendLine(`Reusing existing SmallCoder server on port ${this.port}`);
        return this.port;
      } catch {
        this.output.appendLine('Existing server is not responding, restarting...');
      }
    }

    await this.stopServer();

    const pythonPath = getVenvPython(envPath);
    const scriptPath = path.join(this.extensionPath, 'server', 'server_launcher.py');
    const port = await findFreePort();
    const args = [scriptPath, '--port', `${port}`, '--device', device, '--model-id', modelArg];

    this.output.appendLine(`Starting SmallCoder server with: ${pythonPath} ${args.join(' ')}`);
    this.process = spawn(pythonPath, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    this.port = port;

    if (this.process && this.process.stdout) {
      this.process.stdout.on('data', (chunk) => this.output.appendLine(chunk.toString()));
    }
    if (this.process && this.process.stderr) {
      this.process.stderr.on('data', (chunk) => this.output.appendLine(chunk.toString()));
    }
    if (this.process) {
      this.process.on('exit', (code) => {
        this.output.appendLine(`SmallCoder server exited with code ${code}`);
        if (this.port === port) {
          this.port = null;
          this.currentModel = null;
          this.currentDevice = null;
        }
      });
    }

    await waitForServer(port);
    this.currentModel = modelArg;
    this.currentDevice = device;
    return port;
  }

  public async stopServer(): Promise<void> {
    if (this.process && !this.process.killed) {
      this.output.appendLine('Stopping existing SmallCoder server');
      this.process.kill();
    }
    this.process = null;
    this.port = null;
    this.currentModel = null;
    this.currentDevice = null;
  }
}
