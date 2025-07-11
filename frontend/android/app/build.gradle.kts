plugins {
    id("com.android.application")
    id("kotlin-android")
    // The Flutter Gradle Plugin must be applied after the Android and Kotlin Gradle plugins.
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.seeu" // Or your project's namespace
    compileSdk = 35 // Changed from 34

    defaultConfig {
        applicationId = "com.example.seeu" // Or your app's ID
        minSdk = 24
        targetSdk = 34 // You can keep this at 34 or update it to 35 as well
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }
    // ... rest of your android configuration
    // ... rest of your android configuration

    buildTypes {
        release {
            // TODO: Add your own signing config for the release build.
            // Signing with the debug keys for now, so `flutter run --release` works.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    kotlinOptions {
    jvmTarget = "1.8"
    }
}

flutter {
    source = "../.."
}
