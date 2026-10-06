// Exit teardown under a real Dart VM.
//
// Usage: dart exit_teardown.dart <runtime> <probe> <model> <scenario> <api>
//
// Each scenario leaves the process with a context that no Dart code frees,
// the way a Flutter quit or hot restart does:
//
//   idle             the main isolate loads a model and returns from main
//   leaked           a worker isolate loads a model and is killed; then main
//                    returns: the hot restart case
//   killed-in-load   a worker isolate is killed while it loads, so no Dart
//                    code ever sees the context
//   killed-in-run    a worker isolate is killed while it generates
//   exit-in-run      main calls C exit() while a worker isolate generates,
//                    as a Flutter quit does. dart:io's exit() would not do:
//                    it leaves without running static destructors
//
// With api `tracked` the harness calls sd_dart_new_sd_ctx and
// sd_dart_generate_image, and the process must exit with code 0. With api
// `raw` it calls new_sd_ctx and generate_image: the control, which aborts in
// ggml-metal where Metal residency sets are live.
import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

typedef LoadNative = Pointer<Void> Function(Pointer<Void>);
typedef GenerateNative = Bool Function(
  Pointer<Void>,
  Pointer<Void>,
  Pointer<Pointer<Void>>,
  Pointer<Int>,
);
typedef GenerateDart = bool Function(
  Pointer<Void>,
  Pointer<Void>,
  Pointer<Pointer<Void>>,
  Pointer<Int>,
);

final class Runtime {
  final Pointer<Void> Function(Pointer<Void>) load;
  final GenerateDart generate;
  final Pointer<Void> contextParams;
  final Pointer<Void> generationParams;
  final Pointer<Pointer<Void>> imagesOut;
  final Pointer<Int> countOut;

  Runtime._(
    this.load,
    this.generate,
    this.contextParams,
    this.generationParams,
    this.imagesOut,
    this.countOut,
  );

  factory Runtime(
    String runtimePath,
    String probePath,
    String model,
    bool tracked,
  ) {
    final runtime = DynamicLibrary.open(runtimePath);
    final probe = DynamicLibrary.open(probePath);
    final malloc = DynamicLibrary.process()
        .lookupFunction<
          Pointer<Void> Function(IntPtr),
          Pointer<Void> Function(int)
        >('malloc');
    final path = malloc(model.length + 1).cast<Uint8>();
    for (var i = 0; i < model.length; i++) {
      path[i] = model.codeUnitAt(i);
    }
    path[model.length] = 0;
    probe.lookupFunction<Void Function(), void Function()>('probe_quiet')();
    return Runtime._(
      runtime.lookupFunction<LoadNative, Pointer<Void> Function(Pointer<Void>)>(
        tracked ? 'sd_dart_new_sd_ctx' : 'new_sd_ctx',
      ),
      runtime.lookupFunction<GenerateNative, GenerateDart>(
        tracked ? 'sd_dart_generate_image' : 'generate_image',
      ),
      probe.lookupFunction<LoadNative, Pointer<Void> Function(Pointer<Void>)>(
        'probe_context_params',
      )(path.cast()),
      probe.lookupFunction<
        Pointer<Void> Function(Int),
        Pointer<Void> Function(int)
      >('probe_generation_params')(40),
      malloc(sizeOf<Pointer<Void>>()).cast(),
      malloc(sizeOf<Int>()).cast(),
    );
  }

  Pointer<Void> loadContext() {
    final context = load(contextParams);
    if (context == nullptr) {
      stderr.writeln('exit teardown: the model did not load');
      exit(3);
    }
    return context;
  }

  void generateOn(Pointer<Void> context) =>
      generate(context, generationParams, imagesOut, countOut);
}

void worker((SendPort, List<String>) arguments) {
  final (events, [runtimePath, probePath, model, scenario, api]) = arguments;
  final runtime = Runtime(runtimePath, probePath, model, api == 'tracked');
  events.send('loading');
  final context = runtime.loadContext();
  events.send('loaded');
  if (scenario == 'leaked' || scenario == 'killed-in-load') {
    // Keeps the isolate alive until it is killed.
    ReceivePort();
    return;
  }
  for (;;) {
    events.send('generating');
    runtime.generateOn(context);
  }
}

Future<void> main(List<String> arguments) async {
  final [runtimePath, probePath, model, scenario, api] = arguments;
  if (scenario == 'idle') {
    final runtime = Runtime(runtimePath, probePath, model, api == 'tracked');
    runtime.generateOn(runtime.loadContext());
    return;
  }

  final events = ReceivePort();
  final exited = ReceivePort();
  final isolate = await Isolate.spawn(worker, (
    events.sendPort,
    arguments,
  ), onExit: exited.sendPort);
  final waitFor = switch (scenario) {
    'leaked' => 'loaded',
    'killed-in-load' => 'loading',
    _ => 'generating',
  };
  await events.firstWhere((event) => event == waitFor);
  if (scenario == 'exit-in-run') {
    await Future<void>.delayed(const Duration(milliseconds: 20));
    DynamicLibrary.process()
        .lookupFunction<Void Function(Int), void Function(int)>('exit')(0);
  }
  isolate.kill(priority: Isolate.immediate);
  await exited.first;
  exited.close();
}
