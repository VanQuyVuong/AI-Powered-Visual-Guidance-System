plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.ai_hkdn"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.ai_hkdn"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = false
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
    androidResources {
        // Không nén file model ONNX/manifest để nạp trực tiếp ổn định, không tốn RAM giải nén
        noCompress += listOf("onnx", "json")
    }
}

dependencies {
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)

    // CameraX (Lấy hình trực tiếp từ RAM, không lưu đĩa -> giảm trễ tối đa)
    val cameraxVersion = "1.4.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")

    // ONNX Runtime Android (Hỗ trợ NNAPI FP16 cho NPU Dimensity 7300-Ultra + XNNPACK CPU)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")

    // OkHttp cho WebSocket truyền telemetry & stream
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // Gson parse JSON phản hồi & manifest model
    implementation("com.google.code.gson:gson:2.11.0")

    // Material Icons Extended
    implementation("androidx.compose.material:material-icons-extended")

    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}