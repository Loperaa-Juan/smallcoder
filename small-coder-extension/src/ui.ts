import * as vscode from 'vscode';
import { ModelInfo } from './modelManager';

export async function pickModel(models: ModelInfo[], currentModelId: string): Promise<ModelInfo | undefined> {
  const items = models.map((model) => ({
    label: model.label,
    description: model.description,
    detail: model.id,
    picked: model.id === currentModelId,
    model,
  }));

  const selection = await vscode.window.showQuickPick(items, {
    canPickMany: false,
    placeHolder: 'Choose a SmallCoder model to use for local inference',
  });

  return selection?.model;
}

export function showError(message: string) {
  vscode.window.showErrorMessage(`SmallCoder: ${message}`);
}

export function showInfo(message: string) {
  vscode.window.showInformationMessage(`SmallCoder: ${message}`);
}

export async function withProgress<T>(
  title: string,
  task: (progress: vscode.Progress<{ message?: string; increment?: number }>) => Promise<T>,
) {
  return await vscode.window.withProgress(
    {
      location: vscode.ProgressLocation.Notification,
      title,
      cancellable: false,
    },
    task,
  );
}
