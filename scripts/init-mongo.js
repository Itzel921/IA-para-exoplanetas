// MongoDB Initialization Script
// NASA Space Apps Challenge 2025 - Exoplanet Detection System

// Create database and user
db = db.getSiblingDB('exoplanets_db');

// Create collections with initial indexes
db.createCollection('prediction_results');
db.createCollection('batch_analyses');
db.createCollection('model_metrics');
db.createCollection('user_sessions');

// Create indexes for better performance
print('Creating indexes for prediction_results...');
db.prediction_results.createIndex({ "timestamp": -1 });
db.prediction_results.createIndex({ "prediction": 1 });
db.prediction_results.createIndex({ "confidence": -1 });
db.prediction_results.createIndex({ "user_session": 1 });

print('Creating indexes for batch_analyses...');
db.batch_analyses.createIndex({ "batch_id": 1 }, { unique: true });
db.batch_analyses.createIndex({ "processing_start": -1 });
db.batch_analyses.createIndex({ "status": 1 });

print('Creating indexes for model_metrics...');
db.model_metrics.createIndex({ "model_version": 1 });
db.model_metrics.createIndex({ "evaluation_date": -1 });
db.model_metrics.createIndex({ "accuracy": -1 });

print('Creating indexes for user_sessions...');
db.user_sessions.createIndex({ "session_id": 1 }, { unique: true });
db.user_sessions.createIndex({ "start_time": -1 });
db.user_sessions.createIndex({ "last_activity": -1 });

// Insert initial model metrics (target performance from research)
print('Inserting initial model metrics...');
db.model_metrics.insertOne({
    model_name: "stacking_ensemble",
    model_version: "v1.0",
    accuracy: 0.8308,
    precision: 0.825,
    recall: 0.812,
    f1_score: 0.818,
    roc_auc: 0.948,
    completeness: 0.812,
    reliability: 0.825,
    false_discovery_rate: 0.175,
    test_set_size: 1000,
    positive_class_proportion: 0.25,
    evaluation_date: new Date(),
    hyperparameters: {
        "base_models": ["Random Forest", "AdaBoost", "Extra Trees", "LightGBM"],
        "n_estimators_rf": 1600,
        "learning_rate_ada": 1.0,
        "meta_model": "Logistic Regression"
    },
    feature_count: 789
});

print('MongoDB initialization completed successfully!');
print('Database: exoplanets_db');
print('Collections created: prediction_results, batch_analyses, model_metrics, user_sessions');
print('Target accuracy: 83.08% (Stacking Ensemble)');