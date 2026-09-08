plugins {
    java
}

group = "io.github.jakegilbert7"
version = "0.1.0"

repositories {
    mavenCentral()
    maven {
        name = "papermc"
        url = uri("https://repo.papermc.io/repository/maven-public/")
    }
}

dependencies {
    compileOnly("io.papermc.paper:paper-api:26.2.build.+")
    // Supplied at runtime by Paper's `libraries:` block in plugin.yml, so it is compileOnly
    // here and the jar stays small. Keep the version in both files in step.
    compileOnly("org.java-websocket:Java-WebSocket:1.6.0")
    // Parsing inbound scan requests. Already present in the server's own libraries, so
    // declaring it in plugin.yml costs no extra download.
    compileOnly("com.google.code.gson:gson:2.14.0")
}

java {
    toolchain.languageVersion.set(JavaLanguageVersion.of(25))
}

tasks.register<Copy>("deploy") {
    dependsOn(tasks.jar)
    from(tasks.jar)
    into(file("../server/plugins"))
    rename { "mcgod.jar" }
}