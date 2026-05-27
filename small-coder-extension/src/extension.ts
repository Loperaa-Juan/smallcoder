import * as vscode from 'vscode';

export function activate(context: vscode.ExtensionContext) {
    console.log('Starting up...');

    let disposable = vscode.commands.registerCommand('small-coder-extension.predict', () => {
        vscode.window.showInformationMessage('To be implemented');
    });

    context.subscriptions.push(disposable);
}

export function deactivate() {}
