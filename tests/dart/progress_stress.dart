// Regression harness for progress reporting while the Dart VM shuts down.
//
// Usage: dart progress_stress.dart <library> <design> <owner> <reportMs>
//            <gapUs> <dieAfterMs> <allocatingIsolates>
//
// A worker isolate blocks in a native call that reports progress, as
// generate_image() does. Other isolates allocate, so the isolate group keeps
// requesting safepoints. The isolate that consumes progress (the "owner")
// then goes away while the worker is still reporting:
//
//   owner main        the main isolate owns progress and dies of an error
//   owner background  a background isolate owns it; the main isolate dies
//   owner killed      a background isolate owns it and is killed; then the
//                     main isolate dies
//
// How the owner consumes progress:
//
//   design poll    sd_dart_progress_enable() and a timer that reads
//                  sd_dart_progress_read(); no Dart callback exists
//   design raw     control: a NativeCallable.listener installed with
//                  sd_set_progress_callback()
//   design locked  control: the listener behind a forwarder that locks
//                  around the call, cleared by a NativeFinalizer
//
// The process must end with the unhandled error (exit code 255) once the
// native call returns. The controls instead abort in the VM ("Callback invoked
// after it has been deleted", "GetFfiCallbackMetadata called after shutdown")
// or never exit.
import 'dart:async';
import 'dart:ffi';
import 'dart:io';
import 'dart:isolate';

typedef ProgressNative = Void Function(Int, Int, Float, Pointer<Void>);
typedef SetNative = Void Function(
  Pointer<NativeFunction<ProgressNative>>,
  Pointer<Void>,
);
typedef SetDart = void Function(
  Pointer<NativeFunction<ProgressNative>>,
  Pointer<Void>,
);

final class Progress extends Struct {
  @Uint64()
  external int sequence;
  @Int32()
  external int step;
  @Int32()
  external int steps;
  @Float()
  external double time;
}

final class Anchor implements Finalizable {}

// Kept reachable: a NativeFinalizer that was collected does not run when its
// isolate shuts down.
final anchor = Anchor();
NativeFinalizer? keptFinalizer;
NativeCallable<ProgressNative>? keptCallback;
Timer? keptTimer;

int progressSeen = 0;

void consumeProgress(String libraryPath, String design) {
  final library = DynamicLibrary.open(libraryPath);
  if (design == 'poll') {
    library.lookupFunction<Void Function(), void Function()>(
      'sd_dart_progress_enable',
    )();
    final read = library
        .lookupFunction<
          Void Function(Pointer<Progress>),
          void Function(Pointer<Progress>)
        >('sd_dart_progress_read', isLeaf: true);
    final progress = DynamicLibrary.process()
        .lookupFunction<
          Pointer<Progress> Function(IntPtr),
          Pointer<Progress> Function(int)
        >('malloc')(sizeOf<Progress>());
    var lastSequence = 0;
    keptTimer = Timer.periodic(const Duration(milliseconds: 1), (_) {
      for (var i = 0; i < 64; i++) {
        read(progress);
        final report = progress.ref;
        final whole =
            report.sequence == 0 ||
            (report.steps == report.step + 1 &&
                report.time == report.step % 1024);
        if (!whole || report.sequence < lastSequence) {
          stderr.writeln(
            'stress: inconsistent report ${report.sequence} '
            '${report.step} ${report.steps} ${report.time} after $lastSequence',
          );
          exit(4);
        }
        if (report.sequence > lastSequence) {
          progressSeen++;
        }
        lastSequence = report.sequence;
      }
    });
    return;
  }

  final callback = NativeCallable<ProgressNative>.listener((
    int step,
    int steps,
    double time,
    Pointer<Void> data,
  ) {
    progressSeen++;
  });
  keptCallback = callback;
  if (design == 'raw') {
    library.lookupFunction<SetNative, SetDart>('sd_set_progress_callback')(
      callback.nativeFunction,
      nullptr,
    );
  } else {
    keptFinalizer = NativeFinalizer(
      library.lookup<NativeFinalizerFunction>(
        'control_locked_clear_progress_callback',
      ),
    )..attach(anchor, callback.nativeFunction.cast());
    library.lookupFunction<SetNative, SetDart>(
      'control_locked_set_progress_callback',
    )(callback.nativeFunction, nullptr);
  }
}

void owner((SendPort, String, String) arguments) {
  final (ready, libraryPath, design) = arguments;
  consumeProgress(libraryPath, design);
  final commands = ReceivePort();
  commands.listen((_) => ready.send(progressSeen));
  ready.send(commands.sendPort);
}

void worker((SendPort, String, int, int) arguments) {
  final (started, libraryPath, reportMs, gapUs) = arguments;
  final report = DynamicLibrary.open(libraryPath)
      .lookupFunction<Long Function(Int, Int), int Function(int, int)>(
        'stress_report_progress',
      );
  started.send('reporting');
  report(reportMs, gapUs);
  started.send('reported');
}

void allocate(int seed) {
  var kept = <List<int>>[];
  var value = seed;
  while (true) {
    kept.add(List<int>.filled(256, value++));
    if (kept.length > 2000) {
      kept = <List<int>>[];
    }
  }
}

Future<void> main(List<String> arguments) async {
  final libraryPath = arguments[0];
  final design = arguments[1];
  final ownerLayout = arguments[2];
  final reportMs = int.parse(arguments[3]);
  final gapUs = int.parse(arguments[4]);
  final dieAfter = Duration(milliseconds: int.parse(arguments[5]));
  final allocatingIsolates = int.parse(arguments[6]);

  final messages = ReceivePort();
  final inbox = StreamIterator<Object?>(messages);
  Isolate? ownerIsolate;
  SendPort? ownerCommands;
  if (ownerLayout == 'main') {
    consumeProgress(libraryPath, design);
  } else {
    ownerIsolate = await Isolate.spawn(owner, (
      messages.sendPort,
      libraryPath,
      design,
    ));
    await inbox.moveNext();
    ownerCommands = inbox.current as SendPort;
  }
  for (var i = 0; i < allocatingIsolates; i++) {
    await Isolate.spawn(allocate, i);
  }
  await Isolate.spawn(worker, (
    messages.sendPort,
    libraryPath,
    reportMs,
    gapUs,
  ));
  await inbox.moveNext();

  await Future<void>.delayed(dieAfter);
  var seen = progressSeen;
  if (ownerCommands != null) {
    ownerCommands.send(null);
    await inbox.moveNext();
    if (inbox.current == 'reported') {
      stderr.writeln('stress: the native call returned before the owner died');
      exit(3);
    }
    seen = inbox.current as int;
  }
  if (seen == 0) {
    stderr.writeln('stress: the owner saw no progress');
    exit(5);
  }
  if (ownerLayout == 'killed') {
    final exited = ReceivePort();
    ownerIsolate!.addOnExitListener(exited.sendPort);
    ownerIsolate.kill(priority: Isolate.immediate);
    await exited.first;
  }
  stderr.writeln('stress: dying after $seen progress updates');
  throw StateError('stress: unhandled error');
}
