from sklearn import tree, ensemble, linear_model


models = {
    'decision_tree_entropy': tree.DecisionTreeClassifier(criterion='entropy'),
    'random_forest': ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=20, max_depth=2, criterion='entropy'),
    'sgd_classifier': linear_model.SGDClassifier(loss='log', max_iter=5, n_jobs=-1)
}