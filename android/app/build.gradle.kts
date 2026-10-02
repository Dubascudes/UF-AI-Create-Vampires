plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "edu.ufl.absent"
    compileSdk = 36

    defaultConfig {
        applicationId = "edu.ufl.absent"
        minSdk = 30
        targetSdk = 36
        versionCode = 1
        versionName = "0.1"
    }

    buildTypes {
        // Debuggable builds run with ART optimizations reduced, which roughly doubles the
        // per-pixel conversion time; release is signed with the debug key for local installs.
        release {
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions { jvmTarget = "17" }
    androidResources { noCompress += "tflite" }
    // The NPU loads its Hexagon "skel" libraries from the extracted native library directory.
    packaging { jniLibs { useLegacyPackaging = true } }
}

dependencies {
    implementation("com.google.ar:core:1.56.0")
    implementation("com.google.ai.edge.litert:litert:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu:1.4.2")
    implementation("com.google.ai.edge.litert:litert-gpu-api:1.4.2")
    // Qualcomm's LiteRT delegate and QNN runtime, for LaMa-Dilated on the Hexagon NPU (HTP).
    implementation("com.qualcomm.qti:qnn-litert-delegate:2.50.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.50.0")
}
