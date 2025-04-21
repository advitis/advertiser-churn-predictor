from preprocessing import load_data
from training      import train_model
from evaluation    import evaluate
from scoring       import explain_and_score

def main():
    # 1) Load & split
    X_train, X_test, y_train, y_test = load_data()

    # 2) Train
    model, y_pred = train_model(X_train, y_train, X_test, y_test)

    # 3) Evaluate
    evaluate(y_test, y_pred)

    # 4) Explain & score
    results = explain_and_score(model, X_test)
    print("\nSample churn scores & reasons:")
    print(results.head())

if __name__ == '__main__':
    main()
