import * as vscode from 'vscode';
import * as http from 'http';
import { OutputChannel } from 'vscode';

interface PredictResponse {
  completion: string;
  finish_reason: string;
}

function httpPost<T>(url: string, body: unknown): Promise<T> {
  return new Promise((resolve, reject) => {
    const parsed = new URL(url);
    const data = JSON.stringify(body);
    const request = http.request(
      {
        hostname: parsed.hostname,
        port: parsed.port,
        path: parsed.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(data),
        },
      },
      (res) => {
        let raw = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => {
          raw += chunk;
        });
        res.on('end', () => {
          try {
            resolve(JSON.parse(raw) as T);
          } catch (error) {
            reject(new Error(`Invalid JSON response: ${error}`));
          }
        });
      },
    );

    request.on('error', reject);
    request.write(data);
    request.end();
  });
}

export async function predictAtCursor(port: number, output: OutputChannel) {
  const editor = vscode.window.activeTextEditor;
  if (!editor) {
    vscode.window.showErrorMessage('SmallCoder: No active editor found.');
    return;
  }

  const document = editor.document;
  const cursor = editor.selection.active;
  const startLine = Math.max(0, cursor.line - 4);
  const promptLines = [];

  for (let i = startLine; i <= cursor.line; i += 1) {
    promptLines.push(document.lineAt(i).text);
  }

  const prompt = promptLines.join('\n');
  const language = document.languageId;
  const maxTokens = vscode.workspace.getConfiguration().get<number>('smallCoder.maxTokens', 64);

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'SmallCoder predicting…', cancellable: false },
    async () => {
      const body = {
        prompt,
        cursor_pos: cursor.character,
        language,
        max_tokens: maxTokens,
      };

      output.appendLine(`Sending prediction request to port ${port}`);
      const response = await httpPost<PredictResponse>(`http://127.0.0.1:${port}/predict`, body);
      const completion = response.completion || '';

      await editor.edit((edit) => {
        edit.insert(cursor, completion);
      });
      output.appendLine('SmallCoder prediction inserted');
    },
  );
}
