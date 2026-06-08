import * as vscode from 'vscode';
import * as http from 'http';
import { OutputChannel } from 'vscode';

/**
 * POST a JSON body and stream the chunked text/plain response, invoking
 * `onChunk` for each decoded text piece as it arrives.
 */
function httpPostStream(url: string, body: unknown, onChunk: (text: string) => void): Promise<void> {
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
        res.setEncoding('utf8');
        if (res.statusCode && res.statusCode >= 400) {
          let errorBody = '';
          res.on('data', (chunk) => (errorBody += chunk));
          res.on('end', () => reject(new Error(`Server error ${res.statusCode}: ${errorBody}`)));
          return;
        }
        res.on('data', (chunk: string) => onChunk(chunk));
        res.on('end', () => resolve());
      },
    );

    request.on('error', reject);
    request.write(data);
    request.end();
  });
}

/** Advance a position by the given inserted text, accounting for newlines. */
function advancePosition(pos: vscode.Position, text: string): vscode.Position {
  const newlineIndex = text.lastIndexOf('\n');
  if (newlineIndex === -1) {
    return pos.translate(0, text.length);
  }
  const addedLines = text.split('\n').length - 1;
  const lastLineLength = text.length - newlineIndex - 1;
  return new vscode.Position(pos.line + addedLines, lastLineLength);
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
  const maxTokens = vscode.workspace.getConfiguration().get<number>('smallCoder.maxTokens', 256);

  await vscode.window.withProgress(
    { location: vscode.ProgressLocation.Notification, title: 'SmallCoder predicting…', cancellable: false },
    async () => {
      const body = {
        prompt,
        cursor_pos: cursor.character,
        language,
        max_tokens: maxTokens,
        stream: true,
      };

      output.appendLine(`Sending streaming prediction request to port ${port}`);

      // Insert streamed pieces progressively. Edits are serialized through a
      // promise chain so they never overlap, and grouped into a single undo step.
      let insertPos = cursor;
      let editChain: Promise<void> = Promise.resolve();
      const enqueueInsert = (text: string) => {
        editChain = editChain.then(async () => {
          const at = insertPos;
          await editor.edit((edit) => edit.insert(at, text), { undoStopBefore: false, undoStopAfter: false });
          insertPos = advancePosition(at, text);
        });
      };

      await httpPostStream(`http://127.0.0.1:${port}/predict`, body, enqueueInsert);
      await editChain;
      output.appendLine('SmallCoder streaming prediction complete');
    },
  );
}
