// Exit teardown under a real Dart VM.
//
// Usage: dart exit_teardown.dart <runtime> <probe> <model> <scenario> <api>
//            <backend>|default <image size> <microseconds>
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
//   exit-in-free     main tells a worker isolate to free its context, the
//                    only one, and calls C exit() the given number of
//                    microseconds later: a quit that races a dispose
//
// With api `tracked` the harness calls sd_dart_new_sd_ctx,
// sd_dart_generate_image and sd_dart_exit_free, and the process must exit
// with code 0. With api `raw` it calls new_sd_ctx, generate_image and
// free_sd_ctx: the control, which aborts in ggml-metal where Metal residency
// sets are live.
//
// Either way the library records the runtime's log messages, and the main
// isolate reads them on a timer for as long as it lives, as a binding does:
// through the kill of a worker, C exit() and exit teardown.
import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

typedef LoadNative = Pointer<Void> Function(Pointer<Void>);
typedef ParamsNative = Pointer<Void> Function(Pointer<Void>, Pointer<Void>);
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
typedef LogReadNative = Uint64 Function(
  Uint64,
  Pointer<Void>,
  Size,
  Pointer<Int32>,
  Pointer<Size>,
);
typedef LogReadDart = int Function(
  int,
  Pointer<Void>,
  int,
  Pointer<Int32>,
  Pointer<Size>,
);

const logTextSize = 4096;

final class Runtime {
  final Pointer<Void> Function(Pointer<Void>) load;
  final void Function(Pointer<Void>) free;
  final GenerateDart generate;
  final Pointer<Void> contextParams;
  final Pointer<Void> generationParams;
  final Pointer<Pointer<Void>> imagesOut;
  final Pointer<Int> countOut;
  final LogReadDart readLog;
  final Pointer<Void> logText;
  int logSequence = 0;

  Runtime._(
    this.load,
    this.free,
    this.generate,
    this.contextParams,
    this.generationParams,
    this.imagesOut,
    this.countOut,
    this.readLog,
    this.logText,
  );

  factory Runtime(List<String> arguments) {
    final [runtimePath, probePath, model, _, api, backend, size, _] = arguments;
    final tracked = api == 'tracked';
    final runtime = DynamicLibrary.open(runtimePath);
    final probe = DynamicLibrary.open(probePath);
    final malloc = DynamicLibrary.process()
        .lookupFunction<
          Pointer<Void> Function(IntPtr),
          Pointer<Void> Function(int)
        >('malloc');
    Pointer<Void> native(String text) {
      final bytes = malloc(text.length + 1).cast<Uint8>();
      for (var i = 0; i < text.length; i++) {
        bytes[i] = text.codeUnitAt(i);
      }
      bytes[text.length] = 0;
      return bytes.cast();
    }

    probe.lookupFunction<Void Function(), void Function()>('probe_quiet')();
    return Runtime._(
      runtime.lookupFunction<LoadNative, Pointer<Void> Function(Pointer<Void>)>(
        tracked ? 'sd_dart_new_sd_ctx' : 'new_sd_ctx',
      ),
      runtime.lookupFunction<
        Void Function(Pointer<Void>),
        void Function(Pointer<Void>)
      >(tracked ? 'sd_dart_exit_free' : 'free_sd_ctx'),
      runtime.lookupFunction<GenerateNative, GenerateDart>(
        tracked ? 'sd_dart_generate_image' : 'generate_image',
      ),
      probe.lookupFunction<ParamsNative, ParamsNative>('probe_context_params')(
        native(model),
        native(backend),
      ),
      probe.lookupFunction<
        Pointer<Void> Function(Int, Int),
        Pointer<Void> Function(int, int)
      >('probe_generation_params')(40, int.parse(size)),
      malloc(sizeOf<Pointer<Void>>()).cast(),
      malloc(sizeOf<Int>()).cast(),
      runtime.lookupFunction<LogReadNative, LogReadDart>(
        'sd_dart_log_read',
        isLeaf: true,
      ),
      malloc(logTextSize),
    );
  }

  /// Reads the messages recorded since the last call; returns how many.
  int drainLog() {
    var count = 0;
    for (;;) {
      final next = readLog(logSequence, logText, logTextSize, nullptr, nullptr);
      if (next == 0) {
        return count;
      }
      if (next <= logSequence) {
        stderr.writeln('exit teardown: log sequence $next after $logSequence');
        exit(4);
      }
      logSequence = next;
      count++;
    }
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
  final (events, runtimeArguments) = arguments;
  final scenario = runtimeArguments[3];
  final runtime = Runtime(runtimeArguments);
  events.send('loading');
  final context = runtime.loadContext();
  events.send('loaded');
  if (scenario == 'leaked' || scenario == 'killed-in-load') {
    // Keeps the isolate alive until it is killed.
    ReceivePort();
    return;
  }
  if (scenario == 'exit-in-free') {
    final commands = ReceivePort()..listen((_) => runtime.free(context));
    events.send(commands.sendPort);
    return;
  }
  for (;;) {
    events.send('generating');
    runtime.generateOn(context);
  }
}

Future<void> main(List<String> arguments) async {
  final scenario = arguments[3];
  final runtime = Runtime(arguments);
  if (scenario == 'idle') {
    runtime.generateOn(runtime.loadContext());
    if (runtime.drainLog() == 0) {
      stderr.writeln('exit teardown: a load and a generation logged nothing');
      exit(4);
    }
    return;
  }
  final logPoll = Timer.periodic(
    const Duration(milliseconds: 2),
    (_) => runtime.drainLog(),
  );

  final events = ReceivePort();
  final exited = ReceivePort();
  final isolate = await Isolate.spawn(worker, (
    events.sendPort,
    arguments,
  ), onExit: exited.sendPort);
  final cExit = DynamicLibrary.process()
      .lookupFunction<Void Function(Int), void Function(int)>('exit');
  if (scenario == 'exit-in-free') {
    final commands =
        await events.firstWhere((event) => event is SendPort) as SendPort;
    // Past the time teardown gives a thread after its last call in flight.
    await Future<void>.delayed(const Duration(milliseconds: 400));
    final clock = Stopwatch()..start();
    commands.send(null);
    while (clock.elapsedMicroseconds < int.parse(arguments[7])) {}
    cExit(0);
  }
  final waitFor = switch (scenario) {
    'leaked' => 'loaded',
    'killed-in-load' => 'loading',
    _ => 'generating',
  };
  await events.firstWhere((event) => event == waitFor);
  if (scenario == 'exit-in-run') {
    await Future<void>.delayed(const Duration(milliseconds: 20));
    cExit(0);
  }
  isolate.kill(priority: Isolate.immediate);
  await exited.first;
  exited.close();
  logPoll.cancel();
  runtime.drainLog();
  if (runtime.logSequence == 0) {
    stderr.writeln('exit teardown: the worker logged nothing');
    exit(4);
  }
}
