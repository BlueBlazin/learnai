#!/usr/bin/perl
# Fake language server for tests: speaks the LSP base protocol (Content-Length framing)
# on stdio, so the manager can be exercised without ty.
#
# FAKE_LSP_DIR is a state directory: every start appends a line to `starts`, and the
# pid goes to `server.pid`. With FAKE_LSP_CHILD set, the server also starts a
# `sleep 600` that inherits its stdout and records its pid in `child.pid`.
#
# FAKE_LSP picks the scenario:
#   (empty)    serve, as described below
#   die        print to stderr and exit with code 3 at once, every time
#   die-once   the same on the first start only; serve from the second start on
#   garbage    print text that is not a framed message, then idle
#   deaf       never read stdin
#
# Serving:
#   a request (it has an id and a method) is answered with
#       {"method": ..., "params": ..., "env": {"VIRTUAL_ENV": ..., "cwd": ...}}
#   textDocument/didOpen      pushes textDocument/publishDiagnostics for that document
#   test/exit                 exits with code 3
#   test/big                  sends a notification carrying a 3 MB string
#   test/split                sends one message in two writes, then two in one write
#   test/request              sends a request to the client with id "srv-1"
#   a response (an id and no method) is reported back as a test/response notification
use strict;
use warnings;
use Cwd qw(getcwd);
use IO::Handle;
use JSON::PP;
use Time::HiRes qw(sleep);

binmode(STDIN);
binmode(STDOUT);
STDOUT->autoflush(1);
STDERR->autoflush(1);

my $scenario = $ENV{FAKE_LSP} // "";
my $dir = $ENV{FAKE_LSP_DIR} or die "FAKE_LSP_DIR is not set\n";
my $json = JSON::PP->new->utf8->canonical;

sub put {
    my ($name, $text, $mode) = @_;
    open(my $file, $mode, "$dir/$name") or die "cannot write $dir/$name: $!\n";
    print $file $text;
    close($file);
}

put("starts", "$$\n", ">>");
put("server.pid", "$$\n", ">");
open(my $counter, "<", "$dir/starts") or die;
my $starts = () = <$counter>;
close($counter);

if ($scenario eq "die" || ($scenario eq "die-once" && $starts == 1)) {
    print STDERR "fatal: the fake server was told to die\n";
    exit 3;
}

if ($ENV{FAKE_LSP_CHILD}) {
    my $child = fork();
    if ($child == 0) {
        exec("sleep", "600") or exit 127;
    }
    put("child.pid", "$child\n", ">");
}

if ($scenario eq "deaf") {
    sleep(600);
    exit 0;
}

if ($scenario eq "garbage") {
    print "ty 0.0.0: this line is not a framed message\r\n\r\n";
    sleep(600);
    exit 0;
}

sub frame {
    my $body = $json->encode($_[0]);
    return "Content-Length: " . length($body) . "\r\n\r\n" . $body;
}

sub notify {
    my ($method, $params) = @_;
    print frame({ jsonrpc => "2.0", method => $method, params => $params });
}

print STDERR "fake language server ready\n";
while (1) {
    my $length;
    while (defined(my $line = <STDIN>)) {
        $line =~ s/\r?\n\z//;
        last if $line eq "";
        $length = $1 if $line =~ /^content-length:\s*(\d+)/i;
    }
    exit 0 if eof(STDIN) && !defined $length;
    my $body = "";
    while (length($body) < $length) {
        my $read = read(STDIN, $body, $length - length($body), length($body));
        exit 0 unless $read;
    }
    my $message = $json->decode($body);
    my $method = $message->{method};

    if (defined $method && exists $message->{id}) {
        print frame({
            jsonrpc => "2.0",
            id => $message->{id},
            result => {
                method => $method,
                params => $message->{params},
                env => { VIRTUAL_ENV => $ENV{VIRTUAL_ENV}, cwd => getcwd() },
            },
        });
    } elsif (!defined $method) {
        notify("test/response", $message);
    } elsif ($method eq "textDocument/didOpen") {
        notify("textDocument/publishDiagnostics", {
            uri => $message->{params}{textDocument}{uri},
            diagnostics => [{ message => "fake diagnostic", severity => 1 }],
        });
    } elsif ($method eq "test/exit") {
        print STDERR "fatal: the fake server was asked to exit\n";
        exit 3;
    } elsif ($method eq "test/big") {
        notify("test/big", { text => "x" x (3 * 1024 * 1024) });
    } elsif ($method eq "test/split") {
        my $first = frame({ jsonrpc => "2.0", method => "test/part", params => { n => 1 } });
        my $half = int(length($first) / 2);
        print substr($first, 0, $half);
        sleep(0.05);
        print substr($first, $half);
        print frame({ jsonrpc => "2.0", method => "test/part", params => { n => 2 } })
            . frame({ jsonrpc => "2.0", method => "test/part", params => { n => 3 } });
    } elsif ($method eq "test/request") {
        print frame({
            jsonrpc => "2.0", id => "srv-1", method => "workspace/configuration",
            params => { items => [{ section => "ty" }] },
        });
    }
}
